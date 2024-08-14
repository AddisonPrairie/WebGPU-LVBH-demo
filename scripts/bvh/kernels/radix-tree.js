
function initRadixTreeKernel(device) {
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
                    type: "uniform"
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
    })

    const SM = device.createShaderModule({
        code: SRC(),
        label: "radix tree shader module"
    })

    const PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [BG_LAYOUT]
        }),
        compute: {
            module: SM,
            entryPoint: "compute_radix_tree_pointers"
        }
    })

    return { execute }

    async function execute(KEY_BUFFER, SIZE) {
        if (KEY_BUFFER.size != 4 * SIZE) {
            console.warn(`in radix tree: buffer size [ ${KEY_BUFFER.size} ] does not match requested size [ ${SIZE} ]`)
            return
        }

        // create all the necessary buffers
        const PARENT_BUFFER  = device.createBuffer({
            size: SIZE * 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })
        const UNIFORM_BUFFER = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        })

        // create the necessary bind groups
        const BG = device.createBindGroup({
            layout: BG_LAYOUT,
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: KEY_BUFFER
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    resource: {
                        buffer: UNIFORM_BUFFER
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
        })

        {// send work to GPU
            device.queue.writeBuffer(
                UNIFORM_BUFFER,
                0,
                new Int32Array([
                    SIZE,
                    0,
                    0,
                    0
                ])
            )

            const CE = device.createCommandEncoder()
            const  P = CE.beginComputePass()

            P.setPipeline(PIPELINE)
            P.setBindGroup(0, BG)
            P.dispatchWorkgroups(Math.ceil(SIZE / WG_SIZE))
            P.end()

            device.queue.submit([CE.finish()])
        }

        await device.queue.onSubmittedWorkDone()

        return { PARENT_BUFFER }
    }
    
    function SRC() {
        return /* wgsl */ `

        struct Uniforms {
            num : i32,
            f_1 : i32,
            f_2 : i32,
            f_3 : i32
        };
        
        @group(0) @binding(0) var<storage, read_write> keys : array<u32>;
        @group(0) @binding(1) var<uniform> uniforms : Uniforms;
        @group(0) @binding(2) var<storage, read_write> parents : array<i32>;
        
        @compute @workgroup_size(${WG_SIZE})
        fn compute_radix_tree_pointers(@builtin(global_invocation_id) global_id : vec3u) {
            var idx : i32 = i32(global_id.x);
            if (idx >= uniforms.num - 1) {
                return;
            }

            var pointers : vec2i = compute_child_index(idx);

            // write parent pointer to child nodes, accounting for leaf nodes as well
            if (pointers.x >= 0) {
                parents[pointers.x] = idx;
            } else {
                parents[uniforms.num + - (pointers.x + 1)] = idx;
            }

            if (pointers.y >= 0) {
                parents[pointers.y] = idx;
            } else {
                parents[uniforms.num + - (pointers.y + 1)] = idx;
            }
        }
        
        // computes the first bit (from the most significant) that the two keys differ on
        fn dif(key_1 : u32, key_2 : u32) -> i32 {
            for (var i = 0u; i < 32u; i++) {
                var mask : u32 = 1u << (31u - i);
        
                if ((key_1 & mask) != (key_2 & mask)) {
                    return i32(i);
                }
            }
            return -1;
        }
        
        // computes the length of the common prefix between the keys at idx_1 and idx_2
        fn del(idx_1 : i32, idx_2 : i32) -> i32 {
            // if either index is out of bounds, del() = -1
            if (idx_1 >= uniforms.num || idx_2 >= uniforms.num || idx_1 < 0 || idx_2 < 0) {
                return -1;
            }
        
            var key_dif : i32 = dif(keys[idx_1], keys[idx_2]);
        
            if (key_dif == -1) {
                key_dif = 32 + dif(u32(idx_1), u32(idx_2));
            }
        
            return key_dif;
        }
        
        // computes the index of the left and right child of a given node
        fn compute_child_index(i : i32) -> vec2i {
            // determine the direction of the child range
            var d : i32 = sign(del(i, i + 1) - del(i, i - 1));
        
            // compute a bound on the size of the range
            var del_min : i32 = del(i, i - d);
            var   l_max : i32 = 2;
            while (del(i, i + l_max * d) > del_min) {
                l_max *= 2;
            }
        
            // given this bound, find the true size using binary search
            var l : i32 = 0;
            {
                var t : i32 = l_max / 2;
                while (t > 0) {
                    if (del(i, i + (l + t) * d) > del_min) {
                        l += t;
                    }
                    t /= 2;
                }
            }
            var j : i32 = i + l * d;
        
            // find the split position using binary search
            var del_node : i32 = del(i, j);
            var        s : i32 = 0;
            {
                var v : i32 = 2;
                var t : i32 = (l - 1 + v) / v;
                while (t > 0) {
                    if (del(i, i + (s + t) * d) > del_node) {
                        s += t;
                    }
                    v *= 2;
                    t = (l - 1 + v) / v;
                }
            }
            var gamma : i32 = i + s * d + min(d, 0);
        
            // output (signed) child pointers, where negative indicates leaf node
            var returned : vec2i = vec2i(gamma, gamma + 1);
            if (min(i, j) == gamma) {
                returned.x = -returned.x - 1;
            }
            if (max(i, j) == gamma + 1) {
                returned.y = -returned.y - 1;
            }
        
            return returned;
        }`
    }
}