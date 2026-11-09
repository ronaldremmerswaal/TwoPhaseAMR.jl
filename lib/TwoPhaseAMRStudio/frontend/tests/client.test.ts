import { afterEach, describe, expect, it, vi } from 'vitest';
import { StudioClient } from '../src/client';
import { StudioState } from '../src/state';
import type { PreviewCommand } from '../src/types';

class FakeSocket {
  static readonly OPEN = 1;
  static readonly CONNECTING = 0;
  static instances: FakeSocket[] = [];
  readonly url: string;
  readonly sent: Record<string, unknown>[] = [];
  readyState = FakeSocket.OPEN;
  binaryType = 'blob';
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(url: URL) { this.url = url.toString(); FakeSocket.instances.push(this); }
  send(data: string): void { this.sent.push(JSON.parse(data) as Record<string, unknown>); }
  close(): void { this.readyState = 3; this.onclose?.(); }
  message(data: unknown): void { this.onmessage?.({ data }); }
}

const recording = { active: false, source: null, field: null, path: null, encoded: 0, dropped: 0, status: 'idle' };
const hello = JSON.stringify({ type: 'hello', protocol: 1, runs: [], frames: [], recording });

function response(request: number, value: number): ArrayBuffer {
  const buffer = new ArrayBuffer(36);
  const view = new DataView(buffer);
  [0x54, 0x50, 0x41, 0x53].forEach((byte, index) => view.setUint8(index, byte));
  view.setUint16(4, 1, true);
  view.setUint16(6, 1, true);
  view.setUint32(8, request, true);
  view.setUint8(12, 1);
  view.setUint32(16, 1, true);
  view.setFloat32(32, value, true);
  return buffer;
}

afterEach(() => { vi.unstubAllGlobals(); FakeSocket.instances = []; });

describe('authenticated Studio client', () => {
  it('preserves the token, sends shared commands, and correlates field responses', async () => {
    vi.stubGlobal('WebSocket', FakeSocket);
    const state = new StudioState();
    const client = new StudioClient(state, new URL('https://example.test/?token=abc%26def'));
    client.connect();
    const socket = FakeSocket.instances[0];
    const url = new URL(socket.url);
    expect(url.protocol).toBe('wss:');
    expect(url.pathname).toBe('/ws');
    expect(url.searchParams.get('token')).toBe('abc&def');
    socket.message(hello);
    expect(state.connection).toBe('connected');
    client.control('pause');
    client.getFrames();
    client.recording({ type: 'recording', action: 'start', source: 'stored', path: 'run.mp4', field: 'p', diagnostics: ['mass'] });
    expect(socket.sent.slice(0, 3)).toEqual([
      { type: 'control', action: 'pause' },
      { type: 'get', what: 'frames' },
      { type: 'recording', action: 'start', source: 'stored', path: 'run.mp4', field: 'p', diagnostics: ['mass'] },
    ]);
    const first = client.getField(7, 'p', 'f32', 'liquid');
    const second = client.getField(8, 'p');
    expect(socket.sent[3]).toMatchObject({ type: 'get', what: 'field', frame: 7, field: 'p', phase: 'liquid', request: 1 });
    socket.message(response(2, 22));
    socket.message(response(1, 11));
    expect([...(await first).values]).toEqual([11]);
    expect([...(await second).values]).toEqual([22]);
    const topology = client.getData({ what: 'topology', topology: 9 });
    const planes = client.getData({ what: 'planes', frame: 7 });
    const diagnostics = client.getData({ what: 'diagnostics', names: ['mass'] });
    const steplog = client.getData({ what: 'steplog' });
    expect(socket.sent.slice(5)).toEqual([
      { type: 'get', what: 'topology', topology: 9, request: 3 },
      { type: 'get', what: 'planes', frame: 7, request: 4 },
      { type: 'get', what: 'diagnostics', names: ['mass'], request: 5 },
      { type: 'get', what: 'steplog', request: 6 },
    ]);
    socket.message(JSON.stringify({ type: 'steplog', request: 6, step: [1] }));
    socket.message(JSON.stringify({ type: 'planes', request: 4, frame: 7, phases: [] }));
    socket.message(JSON.stringify({ type: 'diagnostics', request: 5, series: [] }));
    socket.message(JSON.stringify({ type: 'topology', request: 3, id: 9 }));
    expect((await topology).id).toBe(9);
    expect((await planes).frame).toBe(7);
    expect((await diagnostics).series).toEqual([]);
    expect((await steplog).step).toEqual([1]);
  });

  it('correlates preview requests and accepts only a same-origin preview URL', async () => {
    vi.stubGlobal('WebSocket', FakeSocket);
    const state = new StudioState();
    const client = new StudioClient(state, new URL('https://example.test/?token=secret'));
    client.connect();
    const socket = FakeSocket.instances[0];
    socket.message(hello);
    const command: PreviewCommand = {
      type: 'recording', action: 'preview', source: 'stored', frame: 7, field: 'p', diagnostics: [],
      frames: 'all', camera: 'fit', colorrange: 'global', phase_rendering: 'cell',
      size: [640, 360], framerate: 12, every: 1, theme: 'light', arrangement: 'right',
      interface: true, geometry: true, blocks: false,
    };
    const preview = client.preview(command);
    expect(socket.sent[0]).toMatchObject({ action: 'preview', frame: 7, request: 1 });
    socket.message(JSON.stringify({ type: 'recording_preview', request: 1, url: '/preview/abc?token=secret' }));
    await expect(preview).resolves.toBe('https://example.test/preview/abc?token=secret');
    const rejected = expect(client.preview(command)).rejects.toThrow('Studio origin');
    socket.message(JSON.stringify({ type: 'recording_preview', request: 2, url: '/\\evil.test/preview' }));
    await rejected;
  });

  it('delivers request errors and rejects incompatible versions', async () => {
    vi.stubGlobal('WebSocket', FakeSocket);
    const state = new StudioState();
    const client = new StudioClient(state, new URL('http://localhost/?token=secret'));
    client.connect();
    const socket = FakeSocket.instances[0];
    socket.message(hello);
    const request = client.getField(1, 'missing');
    const rejected = expect(request).rejects.toBeInstanceOf(Error);
    socket.message(JSON.stringify({ type: 'error', message: 'Unknown field', request: 1 }));
    await rejected;
    expect(state.error).toBe('Unknown field');
    socket.close();
    client.connect();
    FakeSocket.instances[1].message(JSON.stringify({ type: 'hello', protocol: 2, runs: [], frames: [], recording }));
    expect(state.connection).toBe('error');
    expect(state.error).toBeTruthy();
  });

  it('rejects a response with the wrong data kind for a pending request', async () => {
    vi.stubGlobal('WebSocket', FakeSocket);
    const state = new StudioState();
    const client = new StudioClient(state, new URL('http://localhost/?token=secret'));
    client.connect();
    const socket = FakeSocket.instances[0];
    socket.message(hello);
    const rejected = expect(client.getData({ what: 'planes', frame: 1 })).rejects.toBeInstanceOf(Error);
    socket.message(JSON.stringify({ type: 'topology', request: 1, id: 1 }));
    await rejected;
    expect(state.connection).toBe('error');
  });

  it('refuses a missing token without opening a socket', () => {
    vi.stubGlobal('WebSocket', FakeSocket);
    const state = new StudioState();
    new StudioClient(state, new URL('http://localhost/')).connect();
    expect(FakeSocket.instances).toHaveLength(0);
    expect(state.connection).toBe('error');
  });
});
