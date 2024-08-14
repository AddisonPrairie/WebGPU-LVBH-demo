
function initRadixSortKernel(device) {
    // create bind group layouts
    const SCAN_UP_BG_LAYOUTS = [
        device.createBindGroupLayout({
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
                }
            ]
        }),
        device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }
                }
            ]
        })
    ]

    const INPUT_L_BG_LAYOUTS = [
        device.createBindGroupLayout({
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
                        type: "storage"
                    }
                }
            ]
        }),
        device.createBindGroupLayout({
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
                }
            ]
        }),
        device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }
                }
            ]
        })
    ]

    // compile shaders
    const SCAN_UP_SM = device.createShaderModule({
        code: SCAN_UP_SRC(),
        label: "scan up shader module"
    })

    const INPUT_L_SM = device.createShaderModule({
        code: INPUT_L_SRC(),
        label: "input level shader module"
    })

    // create pipelines
    const INIT_IDX_PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: INPUT_L_BG_LAYOUTS
        }),
        compute: {
            module: INPUT_L_SM,
            entryPoint: "init_idx"
        }
    })

    const INIT_OFF_PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: INPUT_L_BG_LAYOUTS
        }),
        compute: {
            module: INPUT_L_SM,
            entryPoint: "init_off"
        }
    })

    const L_SCAN_PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: INPUT_L_BG_LAYOUTS
        }),
        compute: {
            module: INPUT_L_SM,
            entryPoint: "scan_and_sort"
        }
    })

    const SCAN_UP_PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: SCAN_UP_BG_LAYOUTS
        }),
        compute: {
            module: SCAN_UP_SM,
            entryPoint: "scan_up"
        }
    })

    return { execute }

    // takes as input a buffer of u32's returns a buffer with keys rearranged - is destructive to the buffer!
    async function execute(valBuffer, size) {
        if (valBuffer.size != size * 4) {
            console.warning(`in radix sort: buffer size [ ${valBuffer.size} ] does not match requested size [ ${size} ]`)
            return
        }

        // create all necessary buffers

        const valBuffers = [
            valBuffer,
            device.createBuffer({
                size: valBuffer.size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            })
        ]

        const idxBuffers = [
            device.createBuffer({
                size: valBuffer.size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            }),
            device.createBuffer({
                size: valBuffer.size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            })
        ]

        const l1OffsetsBuffer = device.createBuffer({
            size: 256 * 256 * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })
    
        const l2OffsetsBuffer = device.createBuffer({
            size: 256 * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })
    
        const l3OffsetsBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })
    
        const uniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        })

        // create necessary bind groups

        const SCAN_UP_BGS = [
            device.createBindGroup({
                layout: SCAN_UP_BG_LAYOUTS[0],
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: l1OffsetsBuffer
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: l2OffsetsBuffer
                        }
                    }
                ]
            }),
            device.createBindGroup({
                layout: SCAN_UP_BG_LAYOUTS[0],
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: l2OffsetsBuffer
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: l3OffsetsBuffer
                        }
                    }
                ]
            }),
            device.createBindGroup({
                layout: SCAN_UP_BG_LAYOUTS[1],
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: uniformBuffer
                        }
                    }
                ]
            })
        ]

        const INPUT_L_BGS = [
            device.createBindGroup({
                layout: INPUT_L_BG_LAYOUTS[0],
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: idxBuffers[0]
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: valBuffers[0]
                        }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: idxBuffers[1]
                        }
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: valBuffers[1]
                        }
                    },
                ]
            }),
            device.createBindGroup({
                layout: INPUT_L_BG_LAYOUTS[0],
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: idxBuffers[1]
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: valBuffers[1]
                        }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: idxBuffers[0]
                        }
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: valBuffers[0]
                        }
                    },
                ]
            }),
            device.createBindGroup({
                layout: INPUT_L_BG_LAYOUTS[1],
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: l1OffsetsBuffer
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: l2OffsetsBuffer
                        }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: l3OffsetsBuffer
                        }
                    }
                ]
            }),
        ]

        // initialize the index array
        {
            device.queue.writeBuffer(
                uniformBuffer,
                0,
                new Uint32Array([
                    size,
                    0,
                    0,
                    0
                ])
            )
            
            const CE = device.createCommandEncoder()
            const  P = CE.beginComputePass()

            P.setPipeline(INIT_IDX_PIPELINE)
            P.setBindGroup(0, INPUT_L_BGS[0])
            P.setBindGroup(1, INPUT_L_BGS[2])
            P.setBindGroup(2, SCAN_UP_BGS[2])
            P.dispatchWorkgroups(Math.ceil(size / 256))
            P.end()

            device.queue.submit([CE.finish()])
        }

        // sort the given array based on the 2k, 2k + 1-th bits
        async function sortKthBits(k) {
            {// first pass - update the offsets from the first layer
                const CE = device.createCommandEncoder()

                device.queue.writeBuffer(
                    uniformBuffer,
                    0,
                    new Uint32Array([
                        size,
                        k,
                        0,
                        0
                    ])
                )

                const  P = CE.beginComputePass()
                P.setPipeline(INIT_OFF_PIPELINE)
                P.setBindGroup(0, INPUT_L_BGS[k % 2])
                P.setBindGroup(1, INPUT_L_BGS[2])
                P.setBindGroup(2, SCAN_UP_BGS[2])
                P.dispatchWorkgroups(Math.ceil(size / 256))
                P.end()

                device.queue.submit([CE.finish()])
            }
            {// second pass - scan the level 1 offsets
                const CE = device.createCommandEncoder()

                device.queue.writeBuffer(
                    uniformBuffer,
                    0,
                    new Uint32Array([
                        Math.ceil(size / 256),
                        k,
                        1,
                        0
                    ])
                )

                const  P = CE.beginComputePass()
                P.setPipeline(SCAN_UP_PIPELINE)
                P.setBindGroup(0, SCAN_UP_BGS[0])
                P.setBindGroup(1, SCAN_UP_BGS[2])
                P.dispatchWorkgroups(Math.ceil(size / (256 * 256)))
                P.end()

                device.queue.submit([CE.finish()])
            }
            {// third pass - scan the level 2 offsets
                const CE = device.createCommandEncoder()

                device.queue.writeBuffer(
                    uniformBuffer,
                    0,
                    new Uint32Array([
                        Math.ceil(size / (256 * 256)),
                        k,
                        2,
                        0
                    ])
                )

                const  P = CE.beginComputePass()
                P.setPipeline(SCAN_UP_PIPELINE)
                P.setBindGroup(0, SCAN_UP_BGS[1])
                P.setBindGroup(1, SCAN_UP_BGS[2])
                P.dispatchWorkgroups(1)
                P.end()

                device.queue.submit([CE.finish()])
            }
            {// final pass - scan and write at the first level
                const CE = device.createCommandEncoder()

                device.queue.writeBuffer(
                    uniformBuffer,
                    0,
                    new Uint32Array([
                        size,
                        k,
                        0,
                        0
                    ])
                )

                const P = CE.beginComputePass()
                P.setPipeline(L_SCAN_PIPELINE)
                P.setBindGroup(0, INPUT_L_BGS[k % 2])
                P.setBindGroup(1, INPUT_L_BGS[2])
                P.setBindGroup(2, SCAN_UP_BGS[2])
                P.dispatchWorkgroups(Math.ceil(size / 256))
                P.end()

                device.queue.submit([CE.finish()])
            }
            
            await device.queue.onSubmittedWorkDone()
        }

        // run the 2-bit radix sort 16 times
        for (var k = 0; k < 16; k++) {
            await sortKthBits(k);
        }

        // destroy remaining, unused buffers
        uniformBuffer.destroy()
        valBuffers[1].destroy()
        idxBuffers[1].destroy()
        l1OffsetsBuffer.destroy()
        l2OffsetsBuffer.destroy()
        l3OffsetsBuffer.destroy()

        // return the two key buffers
        return { IDX_BUFFER : idxBuffers[0] }
    }

    function INPUT_L_SRC() {
        return /* wgsl */ `
        // bindgroup specific to interactions with the actual input
        @group(0) @binding(0) var<storage, read_write> idxs : array<i32>;
        @group(0) @binding(1) var<storage, read_write> vals : array<u32>;
        @group(0) @binding(2) var<storage, read_write> n_idxs : array<i32>;
        @group(0) @binding(3) var<storage, read_write> n_vals : array<u32>;

        // bindgroup with counts from intermediate steps
        @group(1) @binding(0) var<storage, read_write> l1_offsets : array<vec4u>;
        @group(1) @binding(1) var<storage, read_write> l2_offsets : array<vec4u>;
        @group(1) @binding(2) var<storage, read_write> l3_offsets : array<vec4u>;

        struct Uniforms {
            num : u32,
            win : u32,
            lvl : u32,
            xtr : u32
        };

        // bindgroup which stores the uniforms
        @group(2) @binding(0) var<uniform> uniforms : Uniforms;

        // set idx in the buffer to just count 0, 1, 2, ...
        @compute @workgroup_size(64)
        fn init_idx(@builtin(global_invocation_id) global_id : vec3u) {
            for (var i : u32 = 0u; i < 4; i++) {
                var idx : u32 = 4u * global_id.x + i;
                if (idx < uniforms.num) {
                    idxs[idx] = i32(idx);
                }
            }
        }

        var<workgroup> wg_count : array<atomic<u32>, 4>;

        // get the number of each element within each group
        @compute @workgroup_size(64)
        fn init_off(
            @builtin(global_invocation_id) global_id : vec3u,
            @builtin(local_invocation_id) local_id : vec3u
        ) {
            // loop over all of this thread's entries and tally how many are of each type
            var l_count : array<u32, 4>;
            for (var i : u32 = 0u; i < 4; i++) {
                var idx : u32 = 4u * global_id.x + i;
                if (idx < uniforms.num) {
                    var value : u32 = vals[idx];
                    l_count[(value >> (2u * uniforms.win)) & 3u]++;
                }
            }

            // send this to workgroup memory
            atomicAdd(&wg_count[0], l_count[0]);
            atomicAdd(&wg_count[1], l_count[1]);
            atomicAdd(&wg_count[2], l_count[2]);
            atomicAdd(&wg_count[3], l_count[3]);

            // the last thread writes the resulting vector to global memory
            workgroupBarrier();
            if (local_id.x == 63u) {
                l1_offsets[global_id.x / 64u] = vec4u(
                    atomicLoad(&wg_count[0]),
                    atomicLoad(&wg_count[1]),
                    atomicLoad(&wg_count[2]),
                    atomicLoad(&wg_count[3])
                );
            }
        }

        var<workgroup> scan_arr : array<vec4u, 64>;

        // scan across the workgroup locally, then reorder everything globally
        @compute @workgroup_size(64)
        fn scan_and_sort(
            @builtin(global_invocation_id) global_id : vec3u,
            @builtin(local_invocation_id) local_id : vec3u
        ) {
            var l_idx : u32 = local_id.x;
            var g_idx : u32 = global_id.x;

            // each thread reads four values from memory and performs a local scan
            var thread_vals : array<u32, 4>;



            for (var i : u32 = 0u; i < 4; i++) {
                var c_idx : u32 = 4u * g_idx + i;
                if (c_idx < uniforms.num) {
                    thread_vals[i] = vals[c_idx];
                }
            }
            
            // compute the offsets across the workgroup
            scan_arr[l_idx] = get_val_vec(thread_vals[0]) 
                            + get_val_vec(thread_vals[1])
                            + get_val_vec(thread_vals[2])
                            + get_val_vec(thread_vals[3]);
            workgroupBarrier();

            workgroup_scan(l_idx);

            // compute the offsets for each element & write to memory
            var thread_offs : array<vec4u, 4>;
            thread_offs[0] = scan_arr[l_idx];
            thread_offs[1] = thread_offs[0] + get_val_vec(thread_vals[0]);
            thread_offs[2] = thread_offs[1] + get_val_vec(thread_vals[1]);
            thread_offs[3] = thread_offs[2] + get_val_vec(thread_vals[2]);

            var global_offsets : vec4u;
            global_offsets[0u] = dot(vec4u(0u, 0u, 0u, 0u), l3_offsets[0u]);
            global_offsets[1u] = dot(vec4u(1u, 0u, 0u, 0u), l3_offsets[0u]);
            global_offsets[2u] = dot(vec4u(1u, 1u, 0u, 0u), l3_offsets[0u]);
            global_offsets[3u] = dot(vec4u(1u, 1u, 1u, 0u), l3_offsets[0u]);

            global_offsets += l1_offsets[g_idx / 64u];
            global_offsets += l2_offsets[g_idx / (64u * 256u)];

            for (var i : u32 = 0u; i < 4; i++) {
                var c_idx : u32 = 4u * g_idx + i;
                if (c_idx < uniforms.num) {
                    var n_idx : u32 = (global_offsets + thread_offs[i])[get_val_u32(thread_vals[i])];

                    n_idxs[n_idx] = idxs[c_idx];
                    n_vals[n_idx] = thread_vals[i];
                }
            }
        }

        // returns which radix index this input is
        fn get_val_u32(input : u32) -> u32 {
            return (input >> (2u * uniforms.win)) & 3u;
        }
        // likewise, but for vector
        fn get_val_vec(input : u32) -> vec4u {
            var shifted = get_val_u32(input);

            if (shifted == 0u) {
                return vec4u(1u, 0u, 0u, 0u);
            }
            if (shifted == 1u) {
                return vec4u(0u, 1u, 0u, 0u);
            }
            if (shifted == 2u) {
                return vec4u(0u, 0u, 1u, 0u);
            }
            
            return vec4u(0u, 0u, 0u, 1u);
        }

        // performs a 256-wide scan on vec4u in scan_arr
        fn workgroup_scan(idx : u32) {
            // upsweep pass
            if ((1u & idx) == 1u) {
                scan_arr[idx] += scan_arr[idx - 1u];
            }
            workgroupBarrier();

            if ((3u & idx) == 3u) {
                scan_arr[idx] += scan_arr[idx - 2u];
            }
            workgroupBarrier();

            if ((7u & idx) == 7u) {
                scan_arr[idx] += scan_arr[idx - 4u];
            }
            workgroupBarrier();

            if ((15u & idx) == 15u) {
                scan_arr[idx] += scan_arr[idx - 8u];
            }
            workgroupBarrier();

            if ((31u & idx) == 31u) {
                scan_arr[idx] += scan_arr[idx - 16u];
            }
            workgroupBarrier();

            // two special cases in transition from upsweep to downsweep
            if (idx == 63u) {
                scan_arr[idx] = scan_arr[31u];
            }
            workgroupBarrier();

            if (idx == 31u) {
                scan_arr[idx] = vec4u(0u);
            }
            workgroupBarrier();

            // downsweep pass
            if ((15u & idx) == 15u && (idx & 16u) != 0u) {
                scan_arr[idx] = scan_arr[idx] + scan_arr[idx - 16u];
            }
            workgroupBarrier();

            if ((15u & idx) == 15u && (idx & 16u) == 0u) {
                scan_arr[idx] = scan_arr[idx + 16u] - scan_arr[idx];
            }
            workgroupBarrier();

            if ((7u & idx) == 7u && (idx & 8u) != 0u) {
                scan_arr[idx] = scan_arr[idx] + scan_arr[idx - 8u];
            }
            workgroupBarrier();

            if ((7u & idx) == 7u && (idx & 8u) == 0u) {
                scan_arr[idx] = scan_arr[idx + 8u] - scan_arr[idx];
            }
            workgroupBarrier();

            if ((3u & idx) == 3u && (idx & 4u) != 0u) {
                scan_arr[idx] = scan_arr[idx] + scan_arr[idx - 4u];
            }
            workgroupBarrier();

            if ((3u & idx) == 3u && (idx & 4u) == 0u) {
                scan_arr[idx] = scan_arr[idx + 4u] - scan_arr[idx];
            }
            workgroupBarrier();

            if ((1u & idx) == 1u && (idx & 2u) != 0u) {
                scan_arr[idx] = scan_arr[idx] + scan_arr[idx - 2u];
            }
            workgroupBarrier();

            if ((1u & idx) == 1u && (idx & 2u) == 0u) {
                scan_arr[idx] = scan_arr[idx + 2u] - scan_arr[idx];
            }
            workgroupBarrier();

            if ((idx & 1u) != 0u) {
                scan_arr[idx] = scan_arr[idx] + scan_arr[idx - 1u];
            }
            workgroupBarrier();

            if ((idx & 1u) == 0u) {
                scan_arr[idx] = scan_arr[idx + 1u] - scan_arr[idx];
            }
            workgroupBarrier();
        }`
    }

    function SCAN_UP_SRC() {
        return /* wgsl */ `
        // bindgroup specific to the intermediate scans
        @group(0) @binding(0) var<storage, read_write> low_count : array<vec4u>;
        @group(0) @binding(1) var<storage, read_write> nex_count : array<vec4u>;

        struct Uniforms {
            num : u32,
            win : u32,
            lvl : u32,
            xtr : u32
        };

        // bindgroup which stores the uniforms
        @group(1) @binding(0) var<uniform> uniforms : Uniforms; 

        // the LDS copy used in the workgroup-wide prefix scan
        var<workgroup> scan_arr : array<vec4u, 64>;

        @compute @workgroup_size(64)
        fn scan_up(
            @builtin(global_invocation_id) global_id : vec3u,
            @builtin(local_invocation_id) local_id : vec3u
        ) {
            var l_idx : u32 = local_id.x;
            var g_idx : u32 = global_id.x;
            
            // each thread reads four values from memory and performs a local scan
            var thread_vals : array<vec4u, 4>;
            var thread_offs : array<vec4u, 4>;

            for (var i : u32 = 0u; i < 4; i++) {
                var c_idx : u32 = 4u * g_idx + i;

                if (c_idx < uniforms.num) {
                    thread_vals[i] = low_count[4u * g_idx + i];
                }
            }

            thread_offs[0] = vec4u(0u, 0u, 0u, 0u);
            thread_offs[1] = thread_vals[0];
            thread_offs[2] = thread_offs[1] + thread_vals[1];
            thread_offs[3] = thread_offs[2] + thread_vals[2];

            // perform the workgroup-wide prefix scan
            scan_arr[l_idx] = thread_vals[0] + thread_vals[1] + thread_vals[2] + thread_vals[3];
            workgroupBarrier();

            workgroup_scan(l_idx);

            // complete the local scan and send it back to storage
            for (var i : u32 = 0u; i < 4; i++) {
                low_count[4u * g_idx + i] = scan_arr[l_idx] + thread_offs[i];
            }

            // if we are the last thread in the group, send the total # to the next layer
            if (l_idx == 63u) {
                nex_count[g_idx / 64u] = scan_arr[63u] + thread_offs[3] + thread_vals[3];
            }
        }

        // performs a 256-wide scan on vec4u in scan_arr
        fn workgroup_scan(idx : u32) {
            // upsweep pass
            if ((1u & idx) == 1u) {
                scan_arr[idx] += scan_arr[idx - 1u];
            }
            workgroupBarrier();

            if ((3u & idx) == 3u) {
                scan_arr[idx] += scan_arr[idx - 2u];
            }
            workgroupBarrier();

            if ((7u & idx) == 7u) {
                scan_arr[idx] += scan_arr[idx - 4u];
            }
            workgroupBarrier();

            if ((15u & idx) == 15u) {
                scan_arr[idx] += scan_arr[idx - 8u];
            }
            workgroupBarrier();

            if ((31u & idx) == 31u) {
                scan_arr[idx] += scan_arr[idx - 16u];
            }
            workgroupBarrier();

            // two special cases in transition from upsweep to downsweep
            if (idx == 63u) {
                scan_arr[idx] = scan_arr[31u];
            }
            workgroupBarrier();

            if (idx == 31u) {
                scan_arr[idx] = vec4u(0u);
            }
            workgroupBarrier();

            // downsweep pass
            if ((15u & idx) == 15u && (idx & 16u) != 0u) {
                scan_arr[idx] = scan_arr[idx] + scan_arr[idx - 16u];
            }
            workgroupBarrier();

            if ((15u & idx) == 15u && (idx & 16u) == 0u) {
                scan_arr[idx] = scan_arr[idx + 16u] - scan_arr[idx];
            }
            workgroupBarrier();

            if ((7u & idx) == 7u && (idx & 8u) != 0u) {
                scan_arr[idx] = scan_arr[idx] + scan_arr[idx - 8u];
            }
            workgroupBarrier();

            if ((7u & idx) == 7u && (idx & 8u) == 0u) {
                scan_arr[idx] = scan_arr[idx + 8u] - scan_arr[idx];
            }
            workgroupBarrier();

            if ((3u & idx) == 3u && (idx & 4u) != 0u) {
                scan_arr[idx] = scan_arr[idx] + scan_arr[idx - 4u];
            }
            workgroupBarrier();

            if ((3u & idx) == 3u && (idx & 4u) == 0u) {
                scan_arr[idx] = scan_arr[idx + 4u] - scan_arr[idx];
            }
            workgroupBarrier();

            if ((1u & idx) == 1u && (idx & 2u) != 0u) {
                scan_arr[idx] = scan_arr[idx] + scan_arr[idx - 2u];
            }
            workgroupBarrier();

            if ((1u & idx) == 1u && (idx & 2u) == 0u) {
                scan_arr[idx] = scan_arr[idx + 2u] - scan_arr[idx];
            }
            workgroupBarrier();

            if ((idx & 1u) != 0u) {
                scan_arr[idx] = scan_arr[idx] + scan_arr[idx - 1u];
            }
            workgroupBarrier();

            if ((idx & 1u) == 0u) {
                scan_arr[idx] = scan_arr[idx + 1u] - scan_arr[idx];
            }
            workgroupBarrier();
        }`
    }
}