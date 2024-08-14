window.onload = async () => {
    const { adapter, device } = await initWebGPU()
    if (!adapter || !device) return

    setBuildTime()
    setParseTime()
    setTriangles()

    const BVH = initBVHBuild(device)
    let PT = null

    let queuedRotate = 0

    async function frame() {
        if (PT) {
            while (queuedRotate > 0) {
                PT.rotateView()
                queuedRotate--
            }
            await PT.sample()
            await PT.sample()
            await PT.sample()
            await PT.draw()
        }

        window.requestAnimationFrame(frame)
    }

    frame()

    // bind all user inputs & UI
    document.querySelector("#rotate-view").addEventListener("mouseup", () => {
        if (PT) queuedRotate++
    })

    function setTriangles(count) {
        let str = ""
        if (count == null) {
            str = "----------"
        } else {
            str = count.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",")
        }
        document.querySelector("#triangle-count").textContent = str
    }

    function setParseTime(time) {
        let str = ""
        if (time == null) {
            str = "----------"
        } else {
            str = time.toString().slice(0, Math.max(time.toString().length, 9)) + "s"
        }
        document.querySelector("#parse-time").textContent = str
    }

    function setBuildTime(time) {
        let str = ""
        if (time == null) {
            str = "----------"
        } else {
            str = time.toString().slice(0, Math.max(time.toString().length, 9)) + "s"
        }
        document.querySelector("#build-time").textContent = str
    }

    {
        async function readFiles(contents) {
            setBuildTime()
            setParseTime()
            setTriangles()
            let s, e
            s = Date.now()
            const { NUM_TRIS, TRI_ARR, BOUNDS } = parseObj(contents[0])

            if (NUM_TRIS > 2_100_000) {
                alert("Warning: Model is too large. Try < 2,000,000 triangles.")
                return
            }

            e = Date.now()
            setParseTime((e - s) / 1000.)
            setTriangles(NUM_TRIS)

            // make thread sleep to update UI
            await new Promise(r => setTimeout(r, 10))

            s = Date.now()
            const { BVH_BUFFER, O_TRIANGLE_BUFFER } = await BVH.build(TRI_ARR, NUM_TRIS, BOUNDS)
            e = Date.now()
            setBuildTime((e - s) / 1000.)
            PT = initPathTracer(device, document.querySelector("#canvas"), {BVH_BUFFER, O_TRIANGLE_BUFFER, BOUNDS})
        }
        
        document.body.addEventListener("drop", (e) => {        
            e.preventDefault()
            e.stopPropagation()
        
            const files = []
            if (e.dataTransfer.items) {
                [...e.dataTransfer.items].forEach((item) => {
                    if (item.kind === "file") {
                        const file = item.getAsFile()
                        if (file.name.endsWith(".obj")) {
                            files.push(file)
                        }
                    }
                })
            } else {
                [...e.dataTransfer.files].forEach((file) => {
                    if (file.name.endsWith('.obj')) {
                        files.push(file)
                    }
                })
            }
        
            // Read all .obj files as text
            const reader = new FileReader()
            const contents = []
            let incr = 0
        
            reader.onload = () => {
                contents.push(reader.result)
                incr++
                if (incr < files.length) {
                    reader.readAsText(files[incr])
                } else {
                    readFiles(contents)
                }
            }
        
            if (files.length > 0) {
                reader.readAsText(files[incr])
            } else {
                alert("File(s) is not valid.")
            }
        })
        
        document.body.addEventListener('dragover', (e) => {
            e.preventDefault()
            e.stopPropagation()
        })
        
        document.body.addEventListener('dragenter', (e) => {
            e.preventDefault()
            e.stopPropagation()
        })
    }
}

async function initWebGPU() {
    const adapter = await navigator.gpu?.requestAdapter()
    const device  = await adapter?.requestDevice()

    if (!device) {
        alert("browser does not support webGPU!")
        return null
    }

    return { adapter, device }
}