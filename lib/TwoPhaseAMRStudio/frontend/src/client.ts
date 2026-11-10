import { decodeField, fieldRequestId, parseServerMessage, type DecodedField } from './protocol';
import { StudioState, validContinuationTarget } from './state';
import type { ControlAction, DataRequest, DataResponse, FieldEncoding, FrameId, Phase, PreviewCommand, RecordingCommand, ServerMessage } from './types';

type PendingRequest =
  | { kind: 'field'; encoding: FieldEncoding; resolve: (field: DecodedField) => void; reject: (error: Error) => void }
  | { kind: DataRequest['what']; resolve: (data: DataResponse) => void; reject: (error: Error) => void }
  | { kind: 'recording_preview'; resolve: (url: string) => void; reject: (error: Error) => void };

function isDataResponse(message: ServerMessage): message is DataResponse {
  return ['topology', 'planes', 'diagnostics', 'steplog'].includes(message.type);
}

/** The only network boundary. Local timeline operations never go through this client. */
export class StudioClient {
  private socket: WebSocket | null = null;
  private pending = new Map<number, PendingRequest>();
  private nextRequest = 1;
  private helloReceived = false;

  constructor(readonly state: StudioState, private readonly pageURL: URL = new URL(window.location.href)) {}

  connect(): void {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) return;
    const token = this.pageURL.searchParams.get('token');
    if (!token) {
      this.state.setConnection('error');
      this.state.setError('A Studio access token is required.');
      return;
    }
    const url = new URL('/ws', this.pageURL);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    url.search = new URLSearchParams({ token }).toString();
    this.state.setConnection('connecting');
    this.helloReceived = false;
    let socket: WebSocket;
    try {
      socket = new WebSocket(url);
    } catch {
      this.state.setConnection('error');
      this.state.setError('Unable to open the Studio connection.');
      return;
    }
    this.socket = socket;
    socket.binaryType = 'arraybuffer';
    socket.onmessage = (event) => {
      try {
        if (typeof event.data === 'string') {
          const message = parseServerMessage(event.data);
          if (!this.helloReceived && message.type !== 'hello') throw new Error('Studio server did not send a hello');
          if (message.type === 'hello') {
            if (this.helloReceived) throw new Error('Unexpected Studio hello');
            this.helloReceived = true;
          }
          if (message.type === 'recording_preview') {
            const pending = this.pending.get(message.request);
            if (!pending) throw new Error('Unexpected preview response request');
            this.pending.delete(message.request);
            if (pending.kind !== 'recording_preview') {
              const error = new Error('Preview response kind does not match request');
              pending.reject(error);
              throw error;
            }
            const url = new URL(message.url, this.pageURL);
            if (url.origin !== this.pageURL.origin) {
              const error = new Error('Preview URL must use the Studio origin');
              pending.reject(error);
              throw error;
            }
            pending.resolve(url.href);
            return;
          }
          if (isDataResponse(message) && message.request != null) {
            const pending = this.pending.get(message.request);
            if (!pending) throw new Error('Unexpected data response request');
            this.pending.delete(message.request);
            if (pending.kind !== message.type) {
              const error = new Error('Data response kind does not match request');
              pending.reject(error);
              throw error;
            }
            pending.resolve(message);
            return;
          }
          if (message.type === 'error' && message.request != null) {
            const pending = this.pending.get(message.request);
            if (pending) {
              this.pending.delete(message.request);
              pending.reject(new Error(message.message));
            }
          }
          this.state.receive(message);
        } else if (event.data instanceof ArrayBuffer) {
          if (!this.helloReceived) throw new Error('Studio server did not send a hello');
          const request = fieldRequestId(event.data);
          const pending = this.pending.get(request);
          if (!pending) throw new Error('Unexpected field response request');
          this.pending.delete(request);
          if (pending.kind !== 'field') {
            const error = new Error('Field response kind does not match request');
            pending.reject(error);
            throw error;
          }
          try { pending.resolve(decodeField(event.data, request, pending.encoding)); }
          catch (error) { pending.reject(error instanceof Error ? error : new Error('Invalid field response')); throw error; }
        } else {
          throw new Error('Unsupported WebSocket message');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Invalid Studio response';
        this.state.setError(message);
        this.state.setConnection('error');
        socket.close(1002, 'Invalid Studio protocol');
      }
    };
    socket.onerror = () => {
      this.state.setConnection('error');
      this.state.setError('Studio connection failed.');
    };
    socket.onclose = () => {
      if (this.socket !== socket) return;
      this.socket = null;
      for (const pending of this.pending.values()) pending.reject(new Error('Studio connection closed'));
      this.pending.clear();
      if (this.state.connection !== 'error') this.state.setConnection('disconnected');
    };
  }

  private send(message: unknown): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN || !this.helloReceived) {
      this.state.setError('Studio is not connected.');
      return false;
    }
    try {
      this.socket.send(JSON.stringify(message));
      return true;
    } catch {
      this.state.setError('Could not send the Studio request.');
      return false;
    }
  }

  control(action: ControlAction): void { this.send({ type: 'control', action }); }
  continueRun(finalTime: number): boolean {
    if (!validContinuationTarget(finalTime, this.state.time)) {
      this.state.setError('The continuation target must be finite and later than the current simulation time.');
      return false;
    }
    return this.send({ type: 'control', action: 'continue', final_time: finalTime });
  }
  getFrames(): void { this.send({ type: 'get', what: 'frames' }); }
  recording(command: RecordingCommand): void { this.send(command); }

  preview(command: PreviewCommand): Promise<string> {
    let request: number;
    try { request = this.allocateRequest(); } catch (error) { return Promise.reject(error); }
    const { promise, resolve, reject } = Promise.withResolvers<string>();
    this.pending.set(request, { kind: 'recording_preview', resolve, reject });
    if (!this.send({ ...command, request })) {
      this.pending.delete(request);
      reject(new Error('Studio is not connected'));
    }
    return promise;
  }

  private allocateRequest(): number {
    let request = this.nextRequest;
    while (this.pending.has(request)) {
      request = request === 0xffff_ffff ? 1 : request + 1;
      if (request === this.nextRequest) throw new Error('No request IDs available');
    }
    this.nextRequest = request === 0xffff_ffff ? 1 : request + 1;
    return request;
  }

  /** Extension panels can request a whole field without owning a socket. */
  getField(frame: FrameId, field: string, encoding: FieldEncoding = 'f32', phase?: Phase): Promise<DecodedField> {
    let request: number;
    try { request = this.allocateRequest(); } catch (error) { return Promise.reject(error); }
    const { promise, resolve, reject } = Promise.withResolvers<DecodedField>();
    this.pending.set(request, { kind: 'field', encoding, resolve, reject });
    const message = { type: 'get', what: 'field', frame, field, encoding, request, ...(phase ? { phase } : {}) };
    if (!this.send(message)) {
      this.pending.delete(request);
      reject(new Error('Studio is not connected'));
    }
    return promise;
  }

  /** Retrieve topology, phase planes, diagnostics, or the step log for a panel. */
  getData(query: DataRequest): Promise<DataResponse> {
    let request: number;
    try { request = this.allocateRequest(); } catch (error) { return Promise.reject(error); }
    const { promise, resolve, reject } = Promise.withResolvers<DataResponse>();
    this.pending.set(request, { kind: query.what, resolve, reject });
    if (!this.send({ type: 'get', ...query, request })) {
      this.pending.delete(request);
      reject(new Error('Studio is not connected'));
    }
    return promise;
  }
}
