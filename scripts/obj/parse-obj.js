//import { OBJFile } from "./obj-file-parser.js"

function parseObj(file) {
    const objFile = new OBJFile(file)
    const output  = objFile.parse()

    let numTris = 0
    let trisArr = []

    let x_min =  1e30
    let y_min =  1e30
    let z_min =  1e30

    let x_max = -1e30
    let y_max = -1e30
    let z_max = -1e30

    for (var x = 0; x < output.models[0].faces.length; x++) {
        let face = output.models[0].faces[x]

        let vr = face.vertices[0]

        let vr_x = output.models[0].vertices[vr.vertexIndex - 1].x
        let vr_z = output.models[0].vertices[vr.vertexIndex - 1].y
        let vr_y = output.models[0].vertices[vr.vertexIndex - 1].z

        x_min = Math.min(x_min, vr_x)
        y_min = Math.min(y_min, vr_y)
        z_min = Math.min(z_min, vr_z)

        x_max = Math.max(x_max, vr_x)
        y_max = Math.max(y_max, vr_y)
        z_max = Math.max(z_max, vr_z)

        for (var y = 1; y < face.vertices.length - 1; y++) {
            let v1 = face.vertices[y + 0]
            let v2 = face.vertices[y + 1]

            let v1_x = output.models[0].vertices[v1.vertexIndex - 1].x
            let v1_z = output.models[0].vertices[v1.vertexIndex - 1].y
            let v1_y = output.models[0].vertices[v1.vertexIndex - 1].z

            let v2_x = output.models[0].vertices[v2.vertexIndex - 1].x
            let v2_z = output.models[0].vertices[v2.vertexIndex - 1].y
            let v2_y = output.models[0].vertices[v2.vertexIndex - 1].z

            x_min = Math.min(v1_x, Math.min(x_min, v2_x))
            y_min = Math.min(v1_y, Math.min(y_min, v2_y))
            z_min = Math.min(v1_z, Math.min(z_min, v2_z))

            x_max = Math.max(v1_x, Math.max(x_max, v2_x))
            y_max = Math.max(v1_y, Math.max(y_max, v2_y))
            z_max = Math.max(v1_z, Math.max(z_max, v2_z))

            trisArr.push(
                vr_x, vr_y, vr_z, 3.1415,
                v1_x, v1_y, v1_z, 3.1415,
                v2_x, v2_y, v2_z, 3.1415,
            )

            numTris++
        }
    }

    // add a floor to the model
    numTris += 2

    let floorHeight = z_min + .01
    let floorSize = 10000.

    trisArr.push(
        -floorSize, -floorSize, floorHeight, 3.1415,
         floorSize, -floorSize, floorHeight, 3.1415,
        -floorSize,  floorSize, floorHeight, 3.1415,
         floorSize, -floorSize, floorHeight, 3.1415,
         floorSize,  floorSize, floorHeight, 3.1415,
        -floorSize,  floorSize, floorHeight, 3.1415,
    )

    return { 
        NUM_TRIS: numTris, 
        TRI_ARR: trisArr, 
        BOUNDS: {
            min: [x_min, y_min, z_min],
            max: [x_max, y_max, z_max]
        } 
    }
}