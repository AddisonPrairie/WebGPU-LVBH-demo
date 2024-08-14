
function initRearrangeKernel(device) {
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
        label: "triangle rearrange shader module"
    })

    const PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [BG_LAYOUT]
        }),
        compute: {
            module: SM,
            entryPoint: "rearrange_triangles"
        }
    })

    return { execute }

    async function execute(I_TRIANGLE_BUFFER, INDEX_BUFFER, size) {
        if (I_TRIANGLE_BUFFER.size != 48 * size) {
            console.warn(`in rearrange: triangle buffer size [ ${I_TRIANGLE_BUFFER.size} ] does not match requested size [ ${size} ]`)
            return
        }
        if (INDEX_BUFFER.size != 4 * size) {
            console.warn(`in rearrange: index buffer size [ ${INDEX_BUFFER.size} ] does not match requested size [ ${size} ]`)
            return
        }

        // create all the necessary buffers
        const O_TRIANGLE_BUFFER = device.createBuffer({
            size: I_TRIANGLE_BUFFER.size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })
        const UNIFORM_BUFFER = device.createBuffer({
            size: 16,
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
                        buffer: I_TRIANGLE_BUFFER
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: O_TRIANGLE_BUFFER
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: INDEX_BUFFER
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
            device.queue.writeBuffer(
                UNIFORM_BUFFER,
                0,
                new Int32Array([
                    size,
                    0,
                    0,
                    0
                ]),
                0
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

        return { O_TRIANGLE_BUFFER }
    }

    function SRC() {
        return /* wgsl */ `
        
        struct Triangle {
            v0 : vec3f,
            v1 : vec3f,
            v2 : vec3f
        };

        struct Uniforms {
            num : i32,
            f_1 : i32,
            f_2 : i32,
            f_3 : i32
        };

        @group(0) @binding(0) var<storage, read_write> i_triangles : array<Triangle>;
        @group(0) @binding(1) var<storage, read_write> o_triangles : array<Triangle>;
        @group(0) @binding(2) var<storage, read_write> new_indices : array<i32>;
        @group(0) @binding(3) var<uniform> uniforms : Uniforms;

        @compute @workgroup_size(${WG_SIZE})
        fn rearrange_triangles(@builtin(global_invocation_id) global_id : vec3u) {
            var idx : i32 = i32(global_id.x);
            if (idx >= uniforms.num) {
                return;
            }
            o_triangles[idx] = i_triangles[new_indices[idx]];
        }`
    }
}