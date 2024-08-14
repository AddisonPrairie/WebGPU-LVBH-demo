
export function initBVHUpPassKernel(device) {
    // shader parameters
    const WG_SIZE = 64

    // create bind group layouts, shader module and pipeline
    const BG_LAYOUTS = [
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
                        type: "storage"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }
                },
            ]
        })
    ]

    const SM = device.createShaderModule({
        code: SRC(),
        label: "radix tree shader module"
    })

    const PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: BG_LAYOUTS
        }),
        compute: {
            module: SM,
            entryPoint: "bvh_upward_pass"
        }
    })

    return { execute }

    async function execute(IDX_BUFFER, AABB_BUFFER, PARENT_BUFFER, size) {
        // create all the necessary buffers
        const     BVH_BUFFER = device.createBuffer({
            size: size * 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })
        const UNIFORM_BUFFER = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        })

        const BGS = [
            device.createBindGroup({
                layout: BG_LAYOUTS[0],
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: IDX_BUFFER
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
                            buffer: PARENT_BUFFER
                        }
                    }
                ]
            }),
            device.createBindGroup({
                layout: BG_LAYOUTS[1],
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: BVH_BUFFER
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        resource: {
                            buffer: UNIFORM_BUFFER
                        }
                    }
                ]
            })
        ]

        {// send work to GPU
            device.queue.writeBuffer(
                UNIFORM_BUFFER,
                0,
                new Int32Array([
                    size,
                    0,
                    0,
                    0
                ])
            )

            const CE = device.createCommandEncoder()
            const  P = CE.beginComputePass()

            P.setPipeline(PIPELINE)
            P.setBindGroup(0, BGS[0])
            P.setBindGroup(1, BGS[1])
            P.dispatchWorkgroups(Math.ceil(size / WG_SIZE))
            P.end()

            device.queue.submit([CE.finish()])
        }

        await device.queue.onSubmittedWorkDone()

        return { BVH_BUFFER }
    }

    function SRC() {
        return /* wgsl */ `

        struct BVHNode {
            aabb_l_min_x : atomic<i32>,
            aabb_l_min_y : atomic<i32>,
            aabb_l_min_z : atomic<i32>,
                 l_child : atomic<i32>,
            aabb_l_max_x : atomic<i32>,
            aabb_l_max_y : atomic<i32>,
            aabb_l_max_z : atomic<i32>,
                     f_1 : atomic<i32>, // Used for synchronization
            aabb_r_min_x : atomic<i32>,
            aabb_r_min_y : atomic<i32>,
            aabb_r_min_z : atomic<i32>,
                 r_child : atomic<i32>,
            aabb_r_max_x : atomic<i32>,
            aabb_r_max_y : atomic<i32>,
            aabb_r_max_z : atomic<i32>,
            f_2 : atomic<i32>
        };

        struct AABB {
            min : vec3f,
            max : vec3f
        };

        struct Uniforms {
            num : i32,
            f_1 : i32,
            f_2 : i32,
            f_3 : i32
        };

        @group(0) @binding(0) var<storage, read_write>   idx_arr : array<i32>;
        @group(0) @binding(1) var<storage, read_write>  aabb_arr : array<AABB>;
        @group(0) @binding(2) var<storage, read_write>   par_arr : array<i32>;

        @group(1) @binding(0) var<storage, read_write> bvh : array<BVHNode>;
        @group(1) @binding(1) var<uniform> uniforms : Uniforms;

        @compute @workgroup_size(${WG_SIZE})
        fn bvh_upward_pass(@builtin(global_invocation_id) global_id : vec3u) {    
            var idx : i32 = i32(global_id.x);
            if (idx >= uniforms.num) {
                return;
            }
        
            var bbox : AABB = aabb_arr[idx_arr[idx]];

            // slightly perturb the bounding box position for check on line ~266
            bbox.min -= vec3f(bbox.min == vec3f(0.)) * vec3f(1e-8f);
            bbox.max += vec3f(bbox.max == vec3f(0.)) * vec3f(1e-8f);

            var c_idx : i32 = idx;
            var w_idx : i32 = -(idx + 1);
            var level : i32 = 0;

            var bSkipped : bool = false;
        
            while ((w_idx != 0 || level == 0) && !bSkipped) {
                var p_idx : i32;
                if (level == 0) {
                    p_idx = par_arr[c_idx + uniforms.num];
                } else {
                    p_idx = par_arr[c_idx];
                }

                if (!bSkipped) {
                    var sibling : i32;
                    
                    if (!bSkipped) {
                        sibling = atomicAdd(&bvh[p_idx].f_1, 1);
                    } 

                    if (sibling == 0 && !bSkipped) {
                        atomicStore(&bvh[p_idx].aabb_l_min_x, bitcast<i32>(bbox.min.x));
                        atomicStore(&bvh[p_idx].aabb_l_min_y, bitcast<i32>(bbox.min.y));
                        atomicStore(&bvh[p_idx].aabb_l_min_z, bitcast<i32>(bbox.min.z));
                        atomicStore(&bvh[p_idx].aabb_l_max_x, bitcast<i32>(bbox.max.x));
                        atomicStore(&bvh[p_idx].aabb_l_max_y, bitcast<i32>(bbox.max.y));
                        atomicStore(&bvh[p_idx].aabb_l_max_z, bitcast<i32>(bbox.max.z));
                        atomicStore(&bvh[p_idx].l_child, w_idx);

                        bSkipped = true;
                    }

                    if (sibling != 0 && !bSkipped) {
                        atomicStore(&bvh[p_idx].aabb_r_min_x, bitcast<i32>(bbox.min.x));
                        atomicStore(&bvh[p_idx].aabb_r_min_y, bitcast<i32>(bbox.min.y));
                        atomicStore(&bvh[p_idx].aabb_r_min_z, bitcast<i32>(bbox.min.z));
                        atomicStore(&bvh[p_idx].aabb_r_max_x, bitcast<i32>(bbox.max.x));
                        atomicStore(&bvh[p_idx].aabb_r_max_y, bitcast<i32>(bbox.max.y));
                        atomicStore(&bvh[p_idx].aabb_r_max_z, bitcast<i32>(bbox.max.z));
                        atomicStore(&bvh[p_idx].r_child, w_idx);

                        var l_min : vec3f = vec3f(
                            bitcast<f32>(atomicLoad(&bvh[p_idx].aabb_l_min_x)),
                            bitcast<f32>(atomicLoad(&bvh[p_idx].aabb_l_min_y)),
                            bitcast<f32>(atomicLoad(&bvh[p_idx].aabb_l_min_z))
                        );
                        var l_max : vec3f = vec3f(
                            bitcast<f32>(atomicLoad(&bvh[p_idx].aabb_l_max_x)),
                            bitcast<f32>(atomicLoad(&bvh[p_idx].aabb_l_max_y)),
                            bitcast<f32>(atomicLoad(&bvh[p_idx].aabb_l_max_z))
                        );

                        // don't do anything if the other is not loaded yet
                        if (any(l_min == vec3f(0.)) || any(l_max == vec3f(0.))) {
                            continue;
                        }

                        bbox.min = min(bbox.min, l_min);
                        bbox.max = max(bbox.max, l_max);

                        // Move to parent
                        c_idx = p_idx;
                        w_idx = p_idx;
                        level += 1;
                    }
                }
            }
        }`
    }
}