import { describe, expect, it } from 'vitest';
import { decodeField, fieldRequestId, parseServerMessage } from '../src/protocol';

function packet(dtype: 1 | 2, values: number[], request = 123, scale = 0.5, offset = -2): ArrayBuffer {
  const buffer = new ArrayBuffer(32 + values.length * (dtype === 1 ? 4 : 2));
  const view = new DataView(buffer);
  for (const [i, byte] of [0x54, 0x50, 0x41, 0x53].entries()) view.setUint8(i, byte);
  view.setUint16(4, 1, true);
  view.setUint16(6, 1, true);
  view.setUint32(8, request, true);
  view.setUint8(12, dtype);
  view.setUint32(16, values.length, true);
  view.setFloat32(20, scale, true);
  view.setFloat32(24, offset, true);
  values.forEach((value, i) => dtype === 1 ? view.setFloat32(32 + 4 * i, value, true) : view.setUint16(32 + 2 * i, value, true));
  return buffer;
}

describe('Studio v1 field messages', () => {
  it('decodes little-endian f32 without losing non-finite values', () => {
    const data = packet(1, [1.25, -3.5, NaN, Infinity]);
    const result = decodeField(data, 123, 'f32');
    expect(result.encoding).toBe('f32');
    expect([...result.values]).toEqual([1.25, -3.5, NaN, Infinity]);
    expect(fieldRequestId(data)).toBe(123);
  });

  it('decodes u16 scale, offset, the missing-value sentinel, and empty arrays', () => {
    const result = decodeField(packet(2, [0, 3, 65534, 65535], 0xffff_ffff), 0xffff_ffff, 'u16');
    expect(result.request).toBe(0xffff_ffff);
    expect([...result.values]).toEqual([-2, -0.5, 32765, NaN]);
    expect(decodeField(packet(2, [], 0), 0).values).toHaveLength(0);
  });

  it('rejects truncated data, wrong request, encoding, magic, kind, version and reserved flags', () => {
    expect(() => decodeField(new ArrayBuffer(31), 1)).toThrow();
    expect(() => decodeField(packet(1, [1]), 124)).toThrow();
    expect(() => decodeField(packet(1, [1]), -1)).toThrow();
    expect(() => decodeField(packet(1, [1]), 123, 'u16')).toThrow();
    expect(() => decodeField(packet(1, [1]).slice(0, 35), 123)).toThrow();
    for (const [offset, value] of [[0, 0], [4, 2], [6, 2], [12, 9], [13, 1], [14, 1], [28, 1]] as const) {
      const data = packet(1, [1]);
      new DataView(data).setUint8(offset, value);
      expect(() => decodeField(data, 123)).toThrow();
    }
  });

  it('rejects invalid field scale and offset', () => {
    for (const dtype of [1, 2] as const) {
      for (const [offset, value] of [[20, NaN], [20, Infinity], [20, -1], [24, NaN], [24, Infinity]]) {
        const data = packet(dtype, [1]);
        new DataView(data).setFloat32(offset, value, true);
        expect(() => decodeField(data, 123)).toThrow();
      }
    }
  });
});

describe('Studio JSON messages', () => {
  it('checks the handshake protocol and message shape', () => {
    const hello = { type: 'hello', protocol: 1, runs: [], frames: [], recording: { active: false } };
    expect(parseServerMessage(JSON.stringify(hello)).type).toBe('hello');
    expect(() => parseServerMessage(JSON.stringify({ ...hello, protocol: 2 }))).toThrow();
    expect(() => parseServerMessage('{')).toThrow();
    expect(() => parseServerMessage('{"type":"frames","frames":null}')).toThrow();
    expect(parseServerMessage('{"type":"error","message":"Denied","request":null}').type).toBe('error');
    expect(() => parseServerMessage('{"type":"error","message":"Denied","request":-1}')).toThrow();
    expect(parseServerMessage('{"type":"recording_preview","request":4,"url":"/preview/4"}').type).toBe('recording_preview');
    expect(() => parseServerMessage('{"type":"recording_preview","request":4,"url":"//evil.test/image"}')).toThrow();
    expect(() => parseServerMessage('{"type":"recording_preview","request":-1,"url":"/preview/4"}')).toThrow();
    expect(() => parseServerMessage('{"type":"unknown"}')).toThrow();
    const schedule = {
      type: 'recording_schedule', request: 4, frames: 41, distinct: 3, repeats: 38, sparse: true,
      window: [0, 0.04], fps: 20, duration: 2, spacing: 0.001, sim_per_second: 0.02, first_frame: 1,
    };
    expect(parseServerMessage(JSON.stringify(schedule)).type).toBe('recording_schedule');
    for (const bad of [
      { ...schedule, request: -1 }, { ...schedule, frames: 1.5 }, { ...schedule, distinct: -1 },
      { ...schedule, duration: -1 }, { ...schedule, spacing: Number.NaN },
      { ...schedule, window: [1, 0] }, { ...schedule, sparse: 'yes' }, { ...schedule, first_frame: null },
    ]) {
      expect(() => parseServerMessage(JSON.stringify(bad))).toThrow();
    }
  });

  it('parses continuation metadata and rejects malformed capability or suggestions', () => {
    const state = { type: 'state', run_state: 'completed', recording: { active: false }, resumable: true, suggested_final_time: 4 };
    expect(parseServerMessage(JSON.stringify(state))).toMatchObject({ resumable: true, suggested_final_time: 4 });
    expect(parseServerMessage(JSON.stringify({ ...state, resumable: false, suggested_final_time: null })))
      .toMatchObject({ resumable: false, suggested_final_time: null });
    expect(() => parseServerMessage(JSON.stringify({ ...state, resumable: 'true' }))).toThrow();
    expect(() => parseServerMessage(JSON.stringify({ ...state, resumable: null }))).toThrow();
    expect(() => parseServerMessage(JSON.stringify({ ...state, suggested_final_time: '4' }))).toThrow();
    expect(() => parseServerMessage('{"type":"state","run_state":"completed","recording":{},"resumable":true,"suggested_final_time":1e999}')).toThrow();
  });
});
