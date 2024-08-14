
function initAABB_ZidxKernel(device) {
    // shader parameters
    const WG_SIZE = 64

    // create bind group layout, shader module and pipeline
    const BG_LAYOUT = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform"
                }
            }
        ]
    })

    const SM = device.createShaderModule({
        code: SRC(),
        label: "AABB/Z-index shader module"
    })

    const PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [BG_LAYOUT]
        }),
        compute: {
            module: SM,
            entryPoint: "compute_aabb_z_idx"
        }
    })

    return { execute }

    async function execute(TRIANGLE_BUFFER, size, bounds) {
        if (TRIANGLE_BUFFER.size != 48 * size) {
            console.warn(`in AABB/Z-index: buffer size [ ${TRIANGLE_BUFFER.size} ] does not match requested size [ ${size} ]`)
            return
        }

        // create all the necessary buffers
        const  AABB_BUFFER = device.createBuffer({
            size: size * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        })
        const Z_IDX_BUFFER = device.createBuffer({
            size: size *  4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })
        const UNIFORM_BUFFER = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        })

        // create the bind group
        const BG = device.createBindGroup({
            layout: BG_LAYOUT,
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: TRIANGLE_BUFFER
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: AABB_BUFFER
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: Z_IDX_BUFFER
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: UNIFORM_BUFFER
                    }
                }
            ]
        })

        {// send work to GPU
            const BUFF = new ArrayBuffer(32)
            const   DV = new DataView(BUFF)

            DV.setFloat32( 0, bounds.min[0], true)
            DV.setFloat32( 4, bounds.min[1], true)
            DV.setFloat32( 8, bounds.min[2], true)
            
            DV.setFloat32(16, bounds.max[0], true)
            DV.setFloat32(20, bounds.max[1], true)
            DV.setFloat32(24, bounds.max[2], true)

            DV.setInt32(12, size, true)

            device.queue.writeBuffer(
                UNIFORM_BUFFER,
                0,
                BUFF,
                0,
                32
            )

            const CE = device.createCommandEncoder()
            const  P = CE.beginComputePass()

            P.setPipeline(PIPELINE)
            P.setBindGroup(0, BG)
            P.dispatchWorkgroups(Math.ceil(size / WG_SIZE))
            P.end()

            device.queue.submit([CE.finish()])
        }

        await device.queue.onSubmittedWorkDone()

        return { AABB_BUFFER, Z_IDX_BUFFER }
    }

    function SRC() {
        return /* wgsl */ `

        struct Triangle {
            v0 : vec3f,
            v1 : vec3f,
            v2 : vec3f
        };

        struct AABB {
            min : vec3f,
            max : vec3f
        };

        struct Uniforms {
            min : vec3f,
            num : i32,
            max : vec3f,
            f_1 : i32
        };

        @group(0) @binding(0) var<storage, read_write> triangles : array<Triangle>;
        @group(0) @binding(1) var<storage, read_write> aabbs     : array<AABB>;
        @group(0) @binding(2) var<storage, read_write> z_indexes : array<u32>;
        @group(0) @binding(3) var<uniform> uniforms : Uniforms;

        @compute @workgroup_size(${WG_SIZE})
        fn compute_aabb_z_idx(@builtin(global_invocation_id) global_id : vec3u) {
            var idx : i32 = i32(global_id.x);
            if (idx >= uniforms.num) {
                return;
            }

            var tri : Triangle = triangles[idx];
            
            var box : AABB;
            box.min = min(tri.v0, min(tri.v1, tri.v2));
            box.max = max(tri.v0, max(tri.v1, tri.v2));

            aabbs[idx] = box;

            var cen : vec3f = (box.max + box.min) * .5f;
            var rel : vec3f = (cen - uniforms.min) / (uniforms.max - uniforms.min);
            
            z_indexes[idx] = morton_code(vec3u(rel * 1023.99f));
        }

        fn morton_code(upos : vec3u) -> u32 {
            return split_3(upos.x) | (split_3(upos.y) << 1) | (split_3(upos.z) << 2);
        }
        
        // from: https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
        fn split_3(u : u32) -> u32 {
            var x : u32 = u;
            x = (x | (x << 16)) & 0x030000FFu;
            x = (x | (x <<  8)) & 0x0300F00Fu;
            x = (x | (x <<  4)) & 0x030C30C3u;
            x = (x | (x <<  2)) & 0x09249249u;
            return x;
        }`
    }
}