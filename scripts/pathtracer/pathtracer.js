// path tracer similar to https://github.com/AddisonPrairie/Personal-Site/blob/main/demos/sdf002/relic/script.js
export function initPathTracer(device, canvas, bvh) {
    const CANVAS = initCanvas(device, canvas)

    let  rot = 0.
    let dist  = 1.5 * Math.max(
        Math.max(
            bvh.BOUNDS.max[0] - bvh.BOUNDS.min[0],
            bvh.BOUNDS.max[1] - bvh.BOUNDS.min[1]
        ),
        bvh.BOUNDS.max[2] - bvh.BOUNDS.min[2]
    )

    let lookAt = [
        (bvh.BOUNDS.min[0] + bvh.BOUNDS.max[0]) * .5,
        (bvh.BOUNDS.min[1] + bvh.BOUNDS.max[1]) * .5,
        (bvh.BOUNDS.min[2] + bvh.BOUNDS.max[2]) * .5,
    ]
    let position = [
        lookAt[0] + Math.cos(rot) * dist,
        lookAt[1] + Math.sin(rot) * dist,
        lookAt[2]
    ]
    let bReset  = true
    

    const { VS, FS, CS } = SRC()

    // create textures for passing data between passes
    const oTextures = [
        device.createTexture({size: [CANVAS.w, CANVAS.h], format: "rgba32float", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING}),
        device.createTexture({size: [CANVAS.w, CANVAS.h], format: "rgba32float", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING})
    ]
    const dTextures = [
        device.createTexture({size: [CANVAS.w, CANVAS.h], format: "rgba32float", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING}),
        device.createTexture({size: [CANVAS.w, CANVAS.h], format: "rgba32float", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING})
    ]
    const tTextures = [
        device.createTexture({size: [CANVAS.w, CANVAS.h], format: "rgba32float", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING}),
        device.createTexture({size: [CANVAS.w, CANVAS.h], format: "rgba32float", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING})
    ]
    const bTextures = [
        device.createTexture({size: [CANVAS.w, CANVAS.h], format: "rgba32float", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING}),
        device.createTexture({size: [CANVAS.w, CANVAS.h], format: "rgba32float", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING})
    ]

    const DRAW_SM = device.createShaderModule({
        code: VS + FS
    })

    const DRAW_BG_LAYOUT = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {
                    sampleType: "unfilterable-float", 
                    viewDimension: "2d", 
                    multisampled: false
                }
            }
        ]
    })

    const DRAW_BGS = [
        device.createBindGroup({
            layout: DRAW_BG_LAYOUT, 
            entries: [
                {
                    binding: 0, 
                    resource: tTextures[1].createView()
                }
            ]
        }),
        device.createBindGroup({
            layout: DRAW_BG_LAYOUT, entries: [
                {
                    binding: 0, 
                    resource: tTextures[0].createView()
                }
            ]
        })
    ]

    const DRAW_PIPELINE = device.createRenderPipeline({
        layout: device.createPipelineLayout({bindGroupLayouts: [DRAW_BG_LAYOUT]}),
        vertex: {
            module: DRAW_SM,
            entryPoint: "vs"
        },
        fragment: {
            module: DRAW_SM,
            entryPoint: "fs",
            targets: [
                {
                    format: CANVAS.presentationFormat
                }
            ]
        }
    })

    const PT_I_BG_LAYOUT = device.createBindGroupLayout({
        entries: [
            {
                binding: 0, 
                visibility: GPUShaderStage.COMPUTE, 
                texture: {
                    sampleType: "unfilterable-float", 
                    viewDimension: "2d", 
                    multisampled: false
                }
            },
            {
                binding: 1, 
                visibility: GPUShaderStage.COMPUTE, 
                texture: {
                    sampleType: "unfilterable-float", 
                    viewDimension: "2d", 
                    multisampled: false
                }
            },
            {
                binding: 2, 
                visibility: GPUShaderStage.COMPUTE, 
                texture: {
                    sampleType: "unfilterable-float", 
                    viewDimension: "2d", 
                    multisampled: false
                }
            },
            {
                binding: 3, 
                visibility: GPUShaderStage.COMPUTE, 
                texture: {
                    sampleType: "unfilterable-float", 
                    viewDimension: "2d", 
                    multisampled: false
                }
            }
        ],
        label: "PT_I_BG_LAYOUT"
    })

    const PT_I_BGS = [
        device.createBindGroup({
            layout: PT_I_BG_LAYOUT,
            entries: [
                {
                    binding: 0, 
                    resource: oTextures[0].createView()
                },
                {
                    binding: 1, 
                    resource: dTextures[0].createView()
                },
                {
                    binding: 2, 
                    resource: tTextures[0].createView()
                },
                {
                    binding: 3, 
                    resource: bTextures[0].createView()
                }
            ]
        }),
        device.createBindGroup({
            layout: PT_I_BG_LAYOUT,
            entries: [
                {
                    binding: 0, 
                    resource: oTextures[1].createView()
                },
                {
                    binding: 1, 
                    resource: dTextures[1].createView()
                },
                {
                    binding: 2, 
                    resource: tTextures[1].createView()
                },
                {
                    binding: 3, 
                    resource: bTextures[1].createView()
                }
            ]
        }),
    ]

    const PT_O_BG_LAYOUT = device.createBindGroupLayout({
        entries: [
            {
                binding: 0, 
                visibility: GPUShaderStage.COMPUTE, 
                storageTexture: {
                    format: "rgba32float", 
                    viewDimension: "2d"
                }
            },
            {
                binding: 1, 
                visibility: GPUShaderStage.COMPUTE, 
                storageTexture: {
                    format: "rgba32float", 
                    viewDimension: "2d"
                }
            },
            {
                binding: 2, 
                visibility: GPUShaderStage.COMPUTE, 
                storageTexture: {
                    format: "rgba32float", 
                    viewDimension: "2d"
                }
            },
            {
                binding: 3, 
                visibility: GPUShaderStage.COMPUTE, 
                storageTexture: {
                    format: "rgba32float", 
                    viewDimension: "2d"
                }
            }
        ],
        label: "PT_O_BG_LAYOUTs"
    })

    const PT_O_BGS = [
        device.createBindGroup({
            layout: PT_O_BG_LAYOUT,
            entries: [
                {
                    binding: 0, 
                    resource: oTextures[1].createView()
                },
                {
                    binding: 1, 
                    resource: dTextures[1].createView()
                },
                {
                    binding: 2, 
                    resource: tTextures[1].createView()
                },
                {
                    binding: 3, 
                    resource: bTextures[1].createView()
                }
            ]
        }),
        device.createBindGroup({
            layout: PT_O_BG_LAYOUT,
            entries: [
                {
                    binding: 0, 
                    resource: oTextures[0].createView()
                },
                {
                    binding: 1, 
                    resource: dTextures[0].createView()
                },
                {
                    binding: 2, 
                    resource: tTextures[0].createView()
                },
                {
                    binding: 3, 
                    resource: bTextures[0].createView()
                }
            ]
        })
    ]

    const PT_BVH_BG_LAYOUT = device.createBindGroupLayout({
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
    })

    const PT_BVH_BG = device.createBindGroup({
        layout: PT_BVH_BG_LAYOUT,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: bvh.BVH_BUFFER
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: bvh.O_TRIANGLE_BUFFER
                }
            }
        ]
    })

    const PT_UNI_BG_LAYOUT = device.createBindGroupLayout({
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

    const UNIFORM_BUFFER = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
    })

    const PT_UNI_BG = device.createBindGroup({
        layout: PT_UNI_BG_LAYOUT,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: UNIFORM_BUFFER
                }
            }
        ]
    })

    const PT_SM = device.createShaderModule({
        code: CS
    })

    const PT_PIPELINE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [PT_I_BG_LAYOUT, PT_O_BG_LAYOUT, PT_BVH_BG_LAYOUT, PT_UNI_BG_LAYOUT]
        }),
        compute: {
            module: PT_SM,
            entryPoint: "main"
        }
    })

    // some variables needed by the methods below
    let ctr = 0

    return { draw, sample, rotateView }

    async function sample() {
        if (bReset) ctr = 0
        if (ctr > 1024) return

        const PP_IDX = ctr++ % 2

        device.queue.writeBuffer(
            UNIFORM_BUFFER,
            0,
            new Float32Array([
                position[0], position[1], position[2], bReset ? 1 : 0,
                lookAt[0], lookAt[1], lookAt[2], 0.
            ]),
            0
        )

        // set reset flag to false so that we don't perpetually re-render
        bReset = false

        const CE = device.createCommandEncoder()
        const  P = CE.beginComputePass()
        P.setPipeline(PT_PIPELINE)
        P.setBindGroup(0, PT_I_BGS[PP_IDX])
        P.setBindGroup(1, PT_O_BGS[PP_IDX])
        P.setBindGroup(2, PT_BVH_BG)
        P.setBindGroup(3, PT_UNI_BG)
        P.dispatchWorkgroups(Math.ceil(CANVAS.w / 8), Math.ceil(CANVAS.h / 8))
        P.end()

        device.queue.submit([CE.finish()])

        await device.queue.onSubmittedWorkDone()

        return
    }

    async function draw() {
        const PP_IDX = ctr % 2

        const CE = device.createCommandEncoder()
        const  P = CE.beginRenderPass({
            colorAttachments: [
                {
                    view: CANVAS.ctx.getCurrentTexture().createView(),
                    clearValue: {r: 1., g: 0., b: 0., a: 1.},
                    loadOp: "clear", 
                    storeOp: "store"
                }
            ]
        })
        P.setPipeline(DRAW_PIPELINE)
        P.setBindGroup(0, DRAW_BGS[PP_IDX])
        P.draw(6)
        P.end()

        device.queue.submit([CE.finish()])

        await device.queue.onSubmittedWorkDone()

        return
    }

    function rotateView() {
        bReset = true
        rot += Math.PI / 4
        position = [
            lookAt[0] + Math.cos(rot) * dist,
            lookAt[1] + Math.sin(rot) * dist,
            lookAt[2]
        ]
    }

    function SRC() {
        let CS = /* wgsl */ `
        @group(0) @binding(0) var otex : texture_2d<f32>;
        @group(0) @binding(1) var dtex : texture_2d<f32>;
        @group(0) @binding(2) var ttex : texture_2d<f32>;
        @group(0) @binding(3) var btex : texture_2d<f32>;

        @group(1) @binding(0) var oout : texture_storage_2d<rgba32float, write>;
        @group(1) @binding(1) var dout : texture_storage_2d<rgba32float, write>;
        @group(1) @binding(2) var tout : texture_storage_2d<rgba32float, write>;
        @group(1) @binding(3) var bout : texture_storage_2d<rgba32float, write>;

        struct BVHNode {
            aabb_l_min : vec3f,
               l_child :   i32,
            aabb_l_max : vec3f,
                   f_1 :   i32,
            aabb_r_min : vec3f,
               r_child :   i32,
            aabb_r_max : vec3f,
                   f_2 :   i32
        };

        struct Triangle {
            v0 : vec3f,
            v1 : vec3f,
            v2 : vec3f
        };

        @group(2) @binding(0) var<storage, read_write> bvh : array<BVHNode >;
        @group(2) @binding(1) var<storage, read_write> tri : array<Triangle>;

        struct Uniforms {
            pos : vec3f,
            rst :   f32,
            lat : vec3f,
        };

        @group(3) @binding(0) var<uniform> uniforms : Uniforms;
        
        const Pi      = 3.14159265358979323846;
        const InvPi   = 0.31830988618379067154;
        const Inv2Pi  = 0.15915494309189533577;
        const Inv4Pi  = 0.07957747154594766788;
        const PiOver2 = 1.57079632679489661923;
        const PiOver4 = 0.78539816339744830961;
        const Sqrt2   = 1.41421356237309504880;
        
        const sw_f : vec2f = vec2f(${CANVAS.w}., ${CANVAS.h}.);
        const sw_u : vec2u = vec2u(${CANVAS.w}u, ${CANVAS.h}u);

        const     fov :   f32 = 60.f;
        const  sinfov :   f32 = sin(.5 * fov * Pi / 180.f);
        const  aspect :   f32 = ${CANVAS.w / CANVAS.h}f;
    
        const  eps    : f32 = .0001;
    
        const mbounce : f32 = 5.;
    
        struct RayHit {
            norm : vec3f,
            dist :   f32
        };

        var<private> stack : array<i32, 32>;

        fn intersect_bvh(o_in : vec3f, d_in : vec3f) -> RayHit {

            var o : vec3f = o_in;
            var d : vec3f = d_in;

            // (lazy) fix for divide by zero errors - change later
            d += vec3f(abs(d) < vec3f(.00001)) * vec3f(.00001);
        
            var dist : f32   = 1e30f;
            var norm : vec3f = vec3f(0.f);

            var stack_ptr : i32 = 0;
            var  node_idx : i32 = 0;

            while (stack_ptr >= 0) {
                // we are testing against a leaf node
                if (node_idx < 0) {
                    var tr : Triangle = tri[-(node_idx + 1)];

                    var n_dis : vec4f = tri_intersect(o, d, tr);

                    if (n_dis.w > 0.f && n_dis.w < dist) {
                        norm = n_dis.xyz;
                        dist = min(n_dis.w, dist);
                    }

                    stack_ptr -= 1;
                    node_idx = stack[stack_ptr];
                } else {
                    var node : BVHNode = bvh[node_idx];

                    var l_dist : f32 = aabb_intersect(
                        node.aabb_l_min, 
                        node.aabb_l_max,
                        o, d
                    );

                    var r_dist : f32 = aabb_intersect(
                        node.aabb_r_min,
                        node.aabb_r_max,
                        o, d
                    );

                    var l_valid : bool = l_dist != -1e30f && l_dist < dist;
                    var r_valid : bool = r_dist != -1e30f && r_dist < dist;

                    if (l_valid && r_valid) {
                        var f_idx : i32;
                        var c_idx : i32;

                        if (l_dist < r_dist) {
                            c_idx = node.l_child;
                            f_idx = node.r_child;
                        } else {
                            c_idx = node.r_child;
                            f_idx = node.l_child;
                        }

                        stack[stack_ptr] = f_idx;
                        stack_ptr += 1;
                        node_idx = c_idx;
                    } else
                    if (l_valid) {
                        node_idx = node.l_child;
                    } else 
                    if (r_valid) {
                        node_idx = node.r_child;
                    } else {
                        stack_ptr -= 1;
                        node_idx = stack[stack_ptr];
                    }
                }
            }

            var returned : RayHit;

            returned.dist = dist;

            if (dot(d, -norm) > 0.) {
                returned.norm =  norm;
            } else {
                returned.norm =  -norm;
            }

            if (returned.dist == 1e30f) {
                returned.dist = -1.f;
            }

            return returned;
        }

        // from: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
        fn tri_intersect(o : vec3f, d : vec3f, tri : Triangle) -> vec4f {
            var v0v1 : vec3f = tri.v1 - tri.v0;
            var v0v2 : vec3f = tri.v2 - tri.v0;
            var pvec : vec3f = cross(d, v0v2);

            var  det : f32 = dot(v0v1, pvec);

            if (abs(det) < 1e-10) {
                return vec4f(-1.f);
            }

            var i_det : f32   = 1.f / det;

            var  tvec : vec3f = o - tri.v0;

            var u : f32 = dot(tvec, pvec) * i_det;
            
            if (u < 0.f || u > 1.f) {
                return vec4f(-1.f);
            }

            var qvec : vec3f = cross(tvec, v0v1);

            var v : f32 = dot(d, qvec) * i_det;
            if (v < 0.f || u + v > 1.f) {
                return vec4f(-1.f);
            }

            return vec4f(
                normalize(cross(v0v1, v0v2)),
                dot(v0v2, qvec) * i_det
            );
        }
        
        fn aabb_intersect(low : vec3f, high : vec3f, o : vec3f, d : vec3f) -> f32 {
            var iDir = 1. / d;
            var f = (high - o) * iDir; var n = (low - o) * iDir;
            var tmax = max(f, n); var tmin = min(f, n);
            var t0 = max(tmin.x, max(tmin.y, tmin.z));
            var t1 = min(tmax.x, min(tmax.y, tmax.z));
            return select(-1e30, select(t0, -1e30, t1 < 0.), t1 >= t0);
        }
    
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id : vec3u) {
            if (any(global_id.xy >= sw_u)) {return;}
            var coord : vec2i = vec2i(global_id.xy);

            var o : vec4f;
            var d : vec4f;
            var t : vec4f;
            var b : vec4f;

            if (uniforms.rst == 0.) {
                o = textureLoad(otex, coord, 0);
                d = textureLoad(dtex, coord, 0);
                t = textureLoad(ttex, coord, 0);
                b = textureLoad(btex, coord, 0);
            }
    
            ptStep(coord, &o, &d, &b, &t);

            textureStore(oout, coord, o);
            textureStore(dout, coord, d);
            textureStore(tout, coord, t);
            textureStore(bout, coord, b);
        }
        
        fn ptStep(coord : vec2i, oin : ptr<function, vec4f>, din : ptr<function, vec4f>, bin : ptr<function, vec4f>, tin : ptr<function, vec4f>) {
            var o : vec3f = (*oin).xyz;
            var d : vec3f = (*din).xyz;
            var b : vec3f = (*bin).xyz;
    
            var    seed : f32 = (*oin).a;
            var bounces : f32 = (*din).a;
    
            var bNewPath : bool = all(b == vec3f(0.));
            var frame0   : bool = bNewPath && ((*tin).a == 0.);
            if (frame0) {
                seed = f32(baseHash(vec2u(coord))) / f32(0xffffffffu) + .008;
            }
    
            if (bNewPath) {
                getCameraRay(vec2f(coord) + rand2(seed), &o, &d); seed += 2.;
                b = vec3f(1.);
            }
    
            var res : RayHit = intersect_bvh(o, d);
            if (res.dist >= 0.) {
                var o1 : vec3f = normalize(ortho(res.norm));
                var o2 : vec3f = normalize(cross(o1, res.norm));
    
                var wo : vec3f = toLocal(o1, o2, res.norm, -d);
                var wi : vec3f;
                var c  : vec3f;
    
                o = o + d * res.dist;

                c = lambertDiffuse(&seed, &wi, wo, vec3f(.3f));
                //c = ggxSmith(&seed, &wi, wo, vec3f(.33f), .1);
                //c = perfectMirror(&wi, wo, vec3f(.2));

                b *= c;
                o += res.norm * 1.01 * eps;
                d = toWorld(o1, o2, res.norm, wi);
    
                if (bounces > 3) {
                    var q : f32 = max(.05f, 1. - b.y);
                    if (rand2(seed).x < q) {
                        b = vec3f(0.);
                    } else {
                        b /= 1. - q;
                    } seed += 2.;
                }
    
                if (all(b == vec3f(0.))) {
                    *tin += vec4f(0., 0., 0., 1.);
                    bounces = -1.;
                }
            } else {
                *tin += vec4f(b * 8., 1.);
                bounces = -1.;
                b = vec3f(0.);
            }
    
            *oin  = vec4f(o, seed);
            *din  = vec4f(d, bounces + 1.);
            *bin  = vec4f(b, 1.);
        }
    
        fn lambertDiffuse(seed : ptr<function, f32>, wi : ptr<function, vec3f>, wo : vec3f, c : vec3f) -> vec3f {
            *wi = cosineSampleHemisphere(rand2(*seed)); *seed += 2.;
            return pow(c, vec3f(2.2));
        }
    
        fn getCameraRay(coord : vec2f, o : ptr<function, vec3f>, d : ptr<function, vec3f>) {
            var sspace : vec2f = coord / sw_f; sspace = sspace * 2. - vec2f(1.); sspace.y *= -1.;
            var local  : vec3f = vec3f(
                aspect * sspace.x * sinfov,
                1.,
                sspace.y * sinfov
            );
            var forward : vec3f = normalize(vec3f(uniforms.lat - uniforms.pos));
            var   right : vec3f = normalize(vec3f(forward.y, -forward.x, 0.));
            var      up : vec3f = cross(right, forward);

            *o = uniforms.pos;
            *d = toWorld(right, forward, up, normalize(local));
        }
    
        fn ortho(v : vec3<f32>) -> vec3<f32> {
            if (abs(v.x) > abs(v.y)) {
                return vec3<f32>(-v.y, v.x, 0.);
            }
            return  vec3<f32>(0., -v.z, v.y);
        }
    
        fn toLocal(v_x : vec3f, v_y : vec3f, v_z : vec3f, w : vec3f) -> vec3f {
            return vec3f(dot(v_x, w), dot(v_y, w), dot(v_z, w));
        }
        
        fn toWorld(v_x : vec3f, v_y : vec3f, v_z : vec3f, w : vec3f) -> vec3f {
            return v_x * w.x + v_y * w.y + v_z * w.z;
        }
        
        //GPU hashes from: https://www.shadertoy.com/view/XlycWh
        fn baseHash(p : vec2u) -> u32 {
            var p2 : vec2u = 1103515245u*((p >> vec2u(1u))^(p.yx));
            var h32 : u32 = 1103515245u*((p2.x)^(p2.y>>3u));
            return h32^(h32 >> 16u);
        }
        fn rand2(seed : f32) -> vec2f {
            var n : u32 = baseHash(bitcast<vec2u>(vec2f(seed + 1., seed + 2.)));
            var rz : vec2u = vec2u(n, n * 48271u);
            return vec2f(rz.xy & vec2u(0x7fffffffu))/f32(0x7fffffff);
        }
    
        //from: pbrt
        fn cosineSampleHemisphere(r2 : vec2f) -> vec3f {
            var d : vec2f = uniformSampleDisk(r2);
            var z : f32 = sqrt(max(0., 1. - d.x * d.x - d.y * d.y));
            return vec3f(d.xy, z);
        }
        fn uniformSampleDisk(r2 : vec2f) -> vec2f {
            var r : f32 = sqrt(max(r2.x, 0.));
            var theta : f32 = 2. * Pi * r2.y;
            return vec2f(r * cos(theta), r * sin(theta));
        }`

        let VS = /* wgsl */ `
        @vertex
        fn vs(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f {
            switch(vertexIndex) {
                case 0u: {
                    return vec4f(1., 1., 0., 1.);}
                case 1u: {
                    return vec4f(-1., 1., 0., 1.);}
                case 2u: {
                    return vec4f(-1., -1., 0., 1.);}
                case 3u: {
                    return vec4f(1., -1., 0., 1.);}
                case 4u: {
                    return vec4f(1., 1., 0., 1.);}
                case 5u: {
                    return vec4f(-1., -1., 0., 1.);}
                default: {
                    return vec4f(0., 0., 0., 0.);}
            }
        }`

        let FS = /* wgsl */ `
        @group(0) @binding(0) var image : texture_2d<f32>;

        fn lum(z : vec3f) -> f32 {
            return dot(z, vec3f(.2126, .7152, .0722));
        }

        @fragment
        fn fs(@builtin(position) fragCoord : vec4f) -> @location(0) vec4f {
            var raw : vec4f = textureLoad(image, vec2i(fragCoord.xy), 0);
            var col : vec3f = raw.xyz / raw.a;

            // apply reinhard tonemap
            col = col / (1.f + lum(col));
            
            return vec4f(
                pow(col, vec3f(1. / 2.2)),
                1.
            );
        }`

        return { CS, VS, FS }
    }
}


function initCanvas(device, canvas) {
    let ctx = canvas.getContext("webgpu")

    let presentationFormat = navigator.gpu.getPreferredCanvasFormat()
    ctx.configure({device, format: presentationFormat})

    const w = Math.ceil(canvas.clientWidth  * 1.5) 
    const h = Math.ceil(canvas.clientHeight * 1.5) 

    canvas.width  = w
    canvas.height = h

    return {
        ctx, presentationFormat, w, h
    }
}