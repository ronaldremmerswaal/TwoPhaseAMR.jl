using Test
using TwoPhaseAMRStudio

# Decode bytes independently of the implementation under test. Positions here
# are one-based; the on-the-wire protocol specifies little-endian offsets.
_u16(bytes, i) = UInt16(bytes[i]) | (UInt16(bytes[i + 1]) << 8)
_u32(bytes, i) = UInt32(_u16(bytes, i)) | (UInt32(_u16(bytes, i + 2)) << 16)
_f32(bytes, i) = reinterpret(Float32, _u32(bytes, i))

function _check_header(bytes, request, dtype, count)
    @test bytes[1:4] == UInt8[0x54, 0x50, 0x41, 0x53] # TPAS
    @test _u16(bytes, 5) == 1
    @test _u16(bytes, 7) == 1  # field
    @test _u32(bytes, 9) == request
    @test bytes[13] == dtype
    @test bytes[14] == 0
    @test _u16(bytes, 15) == 0
    @test _u32(bytes, 17) == count
    return @test _u32(bytes, 29) == 0
end

@testset "Field wire format, quantization, and malformed input" begin
    request = UInt32(0x12345678)
    values = Float32[-3.25, -0.0, 1.5, 9.125, NaN]
    f32 = encode_field(values, request; encoding = :f32)
    @test length(f32) == 32 + 4length(values)
    _check_header(f32, request, 1, length(values))
    raw_f32 = [_f32(f32, 33 + 4(i - 1)) for i in eachindex(values)]
    @test isequal(raw_f32, values)
    decoded = decode_field(f32)
    @test decoded.request == request
    @test decoded.encoding === :f32
    @test isequal(decoded.values, values)

    u16 = encode_field(values, request; encoding = :u16)
    @test length(u16) == 32 + 2length(values)
    _check_header(u16, request, 2, length(values))
    scale, offset = _f32(u16, 21), _f32(u16, 25)
    @test isfinite(scale) && scale > 0
    @test isfinite(offset)
    codes = [_u16(u16, 33 + 2(i - 1)) for i in eachindex(values)]
    @test codes[end] == 0xffff
    @test all(!=(0xffff), codes[1:(end - 1)])
    independently_decoded = [code == 0xffff ? NaN32 : offset + scale * code for code in codes]
    decoded = decode_field(u16)
    @test decoded.request == request
    @test decoded.encoding === :u16
    @test decoded.scale == scale && decoded.offset == offset
    @test isequal(decoded.values, independently_decoded)
    # Half a quantization bin plus rounding in the Float32 scale and offset.
    bound = Float64(scale) / 2 + 8eps(Float32) * maximum(abs, values[1:(end - 1)])
    @test all(abs(Float64(decoded.values[i]) - values[i]) <= bound for i in 1:4)
    @test isnan(decoded.values[end])

    # Empty, constant and entirely missing data have no nonzero dynamic range.
    for input in (Float32[], Float32[2, 2, 2], Float32[NaN, NaN])
        for encoding in (:f32, :u16)
            packet = encode_field(input, UInt32(7); encoding)
            result = decode_field(packet)
            @test result.request == 7
            @test result.encoding === encoding
            @test isequal(result.values, input)
        end
    end
    for request_id in (UInt32(0), typemax(UInt32))
        @test decode_field(encode_field(Float32[1], request_id)).request == request_id
    end
    @test_throws ArgumentError encode_field(values, -1)
    @test_throws ArgumentError encode_field(values, Int64(typemax(UInt32)) + 1)
    @test_throws ArgumentError encode_field(values, true)


    @test_throws Exception encode_field(values, request; encoding = :invalid)
    @test_throws Exception decode_field(UInt8[])
    @test_throws Exception decode_field(f32[1:31])
    @test_throws Exception decode_field(f32[1:(end - 1)])
    @test_throws Exception decode_field([f32; 0x00])
    for (index, byte) in (
            (1, 0x00), (5, 0x02), (7, 0x02), (13, 0xff),
            (14, 0x01), (15, 0x01), (17, 0x06), (29, 0x01),
        )
        malformed = copy(f32)
        malformed[index] = byte
        @test_throws Exception decode_field(malformed)
    end
    negative_scale = copy(u16)
    negative_scale[21:24] .= UInt8[0x00, 0x00, 0x80, 0xbf] # -1f0
    @test_throws ArgumentError decode_field(negative_scale)
    nan_offset = copy(u16)
    nan_offset[25:28] .= UInt8[0x00, 0x00, 0xc0, 0x7f]
    @test_throws ArgumentError decode_field(nan_offset)
end
