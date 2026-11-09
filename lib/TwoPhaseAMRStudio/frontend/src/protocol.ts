import type { FieldEncoding, ServerMessage } from './types';

export const PROTOCOL_VERSION = 1;
const HEADER_BYTES = 32;
const MAX_UINT32 = 0xffff_ffff;
const littleEndianHost = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export interface DecodedField {
  request: number;
  encoding: FieldEncoding;
  values: Float32Array;
  scale: number;
  offset: number;
}

/** Read a correlation id without accepting a malformed or unsupported header. */
export function fieldRequestId(buffer: ArrayBuffer): number {
  if (buffer.byteLength < HEADER_BYTES) throw new Error('Field message has a truncated header');
  const header = new DataView(buffer);
  if (header.getUint8(0) !== 0x54 || header.getUint8(1) !== 0x50 ||
      header.getUint8(2) !== 0x41 || header.getUint8(3) !== 0x53) {
    throw new Error('Invalid field message magic');
  }
  if (header.getUint16(4, true) !== PROTOCOL_VERSION) throw new Error('Unsupported field protocol version');
  if (header.getUint16(6, true) !== 1) throw new Error('Unsupported binary message kind');
  return header.getUint32(8, true);
}

/** Decode v1 little-endian field data, checking the request and the entire envelope. */
export function decodeField(buffer: ArrayBuffer, expectedRequest: number, expectedEncoding?: FieldEncoding): DecodedField {
  if (!Number.isInteger(expectedRequest) || expectedRequest < 0 || expectedRequest > MAX_UINT32) {
    throw new Error('Expected request must be a uint32');
  }
  const request = fieldRequestId(buffer);
  if (request !== expectedRequest) throw new Error('Field response request does not match');
  const header = new DataView(buffer);
  const dtype = header.getUint8(12);
  const encoding = dtype === 1 ? 'f32' : dtype === 2 ? 'u16' : null;
  if (!encoding) throw new Error('Unsupported field data type');
  if (expectedEncoding && encoding !== expectedEncoding) throw new Error('Field response encoding does not match');
  if (header.getUint8(13) !== 0 || header.getUint16(14, true) !== 0 || header.getUint32(28, true) !== 0) {
    throw new Error('Unsupported field flags or reserved bytes');
  }
  const count = header.getUint32(16, true);
  const scale = header.getFloat32(20, true);
  const offset = header.getFloat32(24, true);
  const bytesPerElement = encoding === 'f32' ? 4 : 2;
  if (buffer.byteLength !== HEADER_BYTES + count * bytesPerElement) throw new Error('Field payload length does not match header');
  if (!Number.isFinite(scale) || scale < 0 || !Number.isFinite(offset)) {
    throw new Error('Invalid field scale or offset');
  }
  if (encoding === 'f32' && littleEndianHost) {
    return { request, encoding, scale, offset, values: new Float32Array(buffer, HEADER_BYTES, count) };
  }
  const values = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    if (encoding === 'f32') {
      values[i] = header.getFloat32(HEADER_BYTES + 4 * i, true);
    } else {
      const value = header.getUint16(HEADER_BYTES + 2 * i, true);
      values[i] = value === 0xffff ? NaN : offset + scale * value;
    }
  }
  return { request, encoding, scale, offset, values };
}

/** Parse the supported JSON messages and reject incompatible handshakes. */
export function parseServerMessage(text: string): ServerMessage {
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error('Invalid JSON message from Studio server'); }
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid server message');
  const message = value as Record<string, unknown>;
  switch (message.type) {
    case 'hello':
      if (message.protocol !== PROTOCOL_VERSION) throw new Error(`Unsupported Studio protocol version: ${String(message.protocol)}`);
      if (!Array.isArray(message.runs) || !Array.isArray(message.frames) || !message.recording) throw new Error('Invalid hello message');
      break;
    case 'state':
      if (typeof message.run_state !== 'string' || !message.recording) throw new Error('Invalid state message');
      break;
    case 'frames':
      if (!Array.isArray(message.frames)) throw new Error('Invalid frames message');
      break;
    case 'frame':
      if ((typeof message.id !== 'number' && typeof message.id !== 'string') ||
          typeof message.time !== 'number' || typeof message.step !== 'number' || typeof message.kind !== 'string') {
        throw new Error('Invalid frame message');
      }
      break;
    case 'topology':
    case 'planes':
    case 'diagnostics':
    case 'steplog':
      if (message.request != null && (typeof message.request !== 'number' || !Number.isInteger(message.request) || message.request < 0 || message.request > MAX_UINT32)) {
        throw new Error('Invalid data request id');
      }
      break;
    case 'error':
      if (typeof message.message !== 'string') throw new Error('Invalid error message');
      if (message.request != null && (typeof message.request !== 'number' || !Number.isInteger(message.request) || message.request < 0 || message.request > MAX_UINT32)) {
        throw new Error('Invalid error request id');
      }
      break;
    default:
      throw new Error('Unsupported Studio message type');
  }
  return value as ServerMessage;
}
