const PROTOCOL_VERSION = UInt16(1)
const _FIELD_KIND = UInt16(1)
const _HEADER_BYTES = 32
const _MISSING_U16 = typemax(UInt16)
const _MAX_U16_VALUE = Int(_MISSING_U16) - 1

@inline function _put16!(bytes, at, value)
    n = UInt16(value)
    @inbounds begin
        bytes[at] = UInt8(n & 0xff)
        bytes[at + 1] = UInt8((n >> 8) & 0xff)
    end
    return nothing
end

@inline function _put32!(bytes, at, value)
    n = UInt32(value)
    @inbounds for k in 0:3
        bytes[at + k] = UInt8((n >> (8k)) & 0xff)
    end
    return nothing
end

@inline _putfloat!(bytes, at, value) = _put32!(bytes, at, reinterpret(UInt32, Float32(value)))
@inline _get16(bytes, at) = UInt16(bytes[at]) | (UInt16(bytes[at + 1]) << 8)
@inline function _get32(bytes, at)
    return UInt32(bytes[at]) | (UInt32(bytes[at + 1]) << 8) |
        (UInt32(bytes[at + 2]) << 16) | (UInt32(bytes[at + 3]) << 24)
end
@inline _getfloat(bytes, at) = reinterpret(Float32, _get32(bytes, at))

function _request_id(request)
    request isa Integer && !(request isa Bool) && 0 <= request <= typemax(UInt32) ||
        throw(ArgumentError("request must be an unsigned 32-bit integer"))
    return UInt32(request)
end

"""
    encode_field(values, request; encoding=:f32) -> Vector{UInt8}

Encode an entire real-valued field as a protocol-v1 binary message. The
32-byte header and payload are little-endian. `request` is a UInt32-range
integer. `:f32` writes Float32 values; `:u16` quantizes finite Float32 values
using the header's scale and offset, reserving `0xffff` for nonfinite values.
Constant and empty fields use zero scale. Arrays are flattened in Julia's
iteration order. The encoding is independent of machine byte order.
"""
function encode_field(values::AbstractArray{<:Real}, request; encoding::Symbol = :f32)
    id = _request_id(request)
    encoding in (:f32, :u16) || throw(ArgumentError("encoding must be :f32 or :u16"))
    count = length(values)
    count <= typemax(UInt32) || throw(ArgumentError("field has more than 2^32-1 elements"))
    stride = encoding === :f32 ? 4 : 2
    count <= (typemax(Int) - _HEADER_BYTES) ÷ stride || throw(ArgumentError("field is too large"))
    bytes = zeros(UInt8, _HEADER_BYTES + stride * count)
    bytes[1], bytes[2], bytes[3], bytes[4] = 0x54, 0x50, 0x41, 0x53
    _put16!(bytes, 5, PROTOCOL_VERSION)
    _put16!(bytes, 7, _FIELD_KIND)
    _put32!(bytes, 9, id)
    bytes[13] = encoding === :f32 ? 0x01 : 0x02
    _put32!(bytes, 17, count)
    if encoding === :f32
        _putfloat!(bytes, 21, 1.0f0)
        for (k, value) in enumerate(values)
            _putfloat!(bytes, _HEADER_BYTES + 1 + 4(k - 1), value)
        end
    else
        lo, hi = Inf32, -Inf32
        for value in values
            x = Float32(value)
            if isfinite(x)
                lo, hi = min(lo, x), max(hi, x)
            end
        end
        offset = isfinite(lo) ? lo : 0.0f0
        scale = if !isfinite(lo) || lo == hi
            0.0f0
        else
            max(Float32((Float64(hi) - Float64(lo)) / _MAX_U16_VALUE), nextfloat(0.0f0))
        end
        _putfloat!(bytes, 21, scale)
        _putfloat!(bytes, 25, offset)
        for (k, value) in enumerate(values)
            x = Float32(value)
            code = if !isfinite(x)
                _MISSING_U16
            elseif iszero(scale)
                UInt16(0)
            else
                UInt16(round(Int, clamp((Float64(x) - Float64(offset)) / Float64(scale), 0, _MAX_U16_VALUE)))
            end
            _put16!(bytes, _HEADER_BYTES + 1 + 2(k - 1), code)
        end
    end
    return bytes
end

"""
    decode_field(bytes) -> (; request, encoding, values, scale, offset)

Validate and decode a protocol-v1 field message. The result contains a UInt32
request id, an encoding symbol, Float32 values, and the Float32 quantization
parameters. UInt16 value `0xffff` decodes to NaN. Invalid magic, version, kind,
flags, reserved fields, dtype, parameters, or message length throw an
`ArgumentError`; trailing bytes are not accepted.
"""
function decode_field(bytes::AbstractVector{UInt8})
    Base.require_one_based_indexing(bytes)
    length(bytes) >= _HEADER_BYTES || throw(ArgumentError("truncated field header"))
    (bytes[1], bytes[2], bytes[3], bytes[4]) == (0x54, 0x50, 0x41, 0x53) ||
        throw(ArgumentError("bad field magic"))
    _get16(bytes, 5) == PROTOCOL_VERSION || throw(ArgumentError("unsupported field protocol version"))
    _get16(bytes, 7) == _FIELD_KIND || throw(ArgumentError("unsupported binary message kind"))
    bytes[14] == 0 && _get16(bytes, 15) == 0 && _get32(bytes, 29) == 0 ||
        throw(ArgumentError("unsupported field flags or reserved bits"))
    dtype = bytes[13]
    dtype in (0x01, 0x02) || throw(ArgumentError("unsupported field dtype"))
    encoding = dtype == 0x01 ? :f32 : :u16
    request = _get32(bytes, 9)
    count = Int(_get32(bytes, 17))
    scale, offset = _getfloat(bytes, 21), _getfloat(bytes, 25)
    isfinite(scale) && scale >= 0 && isfinite(offset) ||
        throw(ArgumentError("invalid field quantization parameters"))
    stride = encoding === :f32 ? 4 : 2
    count <= (typemax(Int) - _HEADER_BYTES) ÷ stride || throw(ArgumentError("field is too large"))
    length(bytes) == _HEADER_BYTES + stride * count || throw(ArgumentError("invalid field payload length"))
    values = Vector{Float32}(undef, count)
    for k in 1:count
        at = _HEADER_BYTES + 1 + stride * (k - 1)
        if encoding === :f32
            values[k] = _getfloat(bytes, at)
        else
            code = _get16(bytes, at)
            if code == _MISSING_U16
                values[k] = NaN32
            else
                value = offset + scale * code
                # Avoid an intermediate Float32 overflow on a very wide range.
                values[k] = isfinite(value) ? value : Float32(Float64(offset) + Float64(scale) * code)
            end
        end
    end
    return (; request, encoding, values, scale, offset)
end
