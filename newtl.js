const tfjs = document.createElement('script')
tfjs.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js'
const p5js = document.createElement('script')
p5js.src = 'https://cdnjs.cloudflare.com/ajax/libs/p5.js/0.8.0/p5.js'

document.head.appendChild(tfjs)
document.head.appendChild(p5js)

class Newt{
    constructor(args){
        //Mendatory args
        this.args = args
        this.generation = 0
        this.agents = []
        this.savedAgents = []
        this.bestFitness = 0
        this.totalFitness = 0
        this.popSize = args.popSize
        for(let i = 0; i < this.popSize; i++)
            this.agents.push(new Agent(args.agentClass, args.netConfig,
                args.inputFetch, args.actionTable))
        this.agents[0].best = true
        this.fitFunc = args.fitnessFunction
        this.reset = args.reset
        //Optionnal args
        this.cps = args.cps ? args.cps : 1
        this.mutationRate = args.mutationRate ? args.mutationRate : 0.1
        this.mutationRateF = args.mRateFunction ? args.mRateFunction : x=>x
        this.backEnd = args.backEnd ? args.backEnd : "cpu"
        this.showBest = args.showBest ? args.showBest : false

        this.renderText = true
        this.renderNet = true
        this.framesInterval = 0
        this.renderer = new Renderer({textBuffer: args.textBuffer, renderBuffer: args.renderBuffer,
            tBX: args.textBufferX, tBY: args.textBufferY, rBX: args.renderBufferX, rBY: args.renderBufferY})

        tf.setBackend(this.backEnd)

        this.renderer.showNet(this.findBestLabel(this.agents).brain)
    }
    update(){
        for(let i = 0; i < this.cps; i++){
            for(let agent of this.agents)
                agent.update()
            
            for(let i = this.agents.length-1; i >= 0; i--)
                if(!this.agents[i].body.alive)
                    this.savedAgents.push(this.agents.splice(i, 1)[0])
                
            if(this.agents.length === 0) this.newGen()
        }
        this.show()
    }
    show(){
        if(!this.showBest)
            for(let agent of this.agents)
                agent.show()
        if(this.showBest)
            this.findBestLabel(this.agents).show()

        if(this.args.textBuffer && this.renderText)
            this.renderer.showText(this)
        
        tint(255, 100)
        if(this.args.renderBuffer && this.renderNet)
            image(this.renderer.buffer2, this.renderer.rBX, this.renderer.rBY)

    }
    newGen(){
        this.mutationRate = this.mutationRateF(this.mutationRate)
        this.calculateFitness()

        const best = this.findBestFitness(this.savedAgents)
        const newBestBody = new Agent(this.args.agentClass, this.args.netConfig,
            this.args.inputFetch, this.args.actionTable, false)
        newBestBody.brain = best.brain.copy()
        this.agents.push(newBestBody)
        this.agents[0].best = true
        for(let i = 0; i < this.popSize-1; i++)
            this.agents.push(this.pickOne(this.savedAgents))
        

        for(let agent of this.savedAgents)
            tf.dispose(agent.brain.model)
        
        this.savedAgents = []
        ++this.generation
        this.reset()

        this.renderer.showNet(this.findBestLabel(this.agents).brain)
    }
    calculateFitness(){
        let sum = 0
        for(let agent of this.savedAgents){
            agent.fitness = this.fitFunc(agent.body)
            sum += agent.fitness
        }
        for(let agent of this.savedAgents)
            agent.fitness = agent.fitness/sum
    }
    pickOne(array){
        let index = 0;
        let r = random(1);
        while (r > 0) {
            r = r - array[index].fitness;
            index++;
        }
        index--;
        let child = array[index];
        let agent = new Agent(this.args.agentClass, this.args.netConfig,
            this.args.inputFetch, this.args.actionTable, false)

        agent.brain = child.brain.copy()
        agent.brain.mutate(this.mutationRate)

        return agent;
    }
    findBestFitness(array){
        let bestFit = {fitness: -Infinity}
        let sum = 0
        for(let agent of array){
            if(agent.fitness > bestFit.fitness) bestFit = agent
            sum+=agent.fitness
        }
        this.bestFitness = bestFit.fitness
        this.totalFitness = sum
        return bestFit
    }
    findBestLabel(array){
        for(let agent of array)
            if(agent.best = true) return agent
        return array[0]
    }
}


class Agent{
    constructor(clss, brainConfig, inputFetch, actionTable, brain = true){
        this.body = new clss()
        this.brain = brain ? new NeuralNetwork(brainConfig.layers_num, brainConfig.layers_act, brainConfig.layers_type) : null
        this.inputFetch = inputFetch
        this.actionTable = actionTable
        this.best = false
        this.fitness = 0
    }

    update(){
        //predict
        let x_arr = this.inputFetch(this.body)
        let thought = this.brain.predict(x_arr)
        let highest = 0
        for(let i = 0; i < thought.length; i++)
            if(thought[i] > highest) highest = thought[i]
        let action = this.actionTable[thought.indexOf(highest)]
        //act
        this.body.update(action)
    }
    show(){
        this.body.show()
    }
}


class NeuralNetwork{

    constructor(layers_num, layers_act, layers_type){
        this.layers_num = layers_num
        this.layers_act = layers_act
        this.layers_type = layers_type

        this.createModel()
    }

    createModel(){
        tf.tidy(()=>{
            this.model = tf.sequential()

            this.layers = []

            for(let i = 1; i < this.layers_num.length; i++){
                let act = this.layers_act[i-1]
                switch(this.layers_type[i-1]){
                    case "lstm":
                    case "LSTM":
                        this.layers.push(tf.layers.lstm({
                            inputShape: i === 1 ? [this.layers_num[i-1]] : undefined,
                            units: this.layers_num[i],
                            activation: act,
                            returnSequences: i < this.layers_num.length-1,
                        }))
                        break;
                    case "dense":
                    default:
                        this.layers.push(tf.layers.dense({
                            inputShape: [this.layers_num[i-1]],
                            units: this.layers_num[i],
                            activation: act
                        }))
                        break;
                }
                
                this.model.add(this.layers[i-1])
            }
        })
    }

    predict(arr){
        for(let i = 0; i < arr.length; i++)
            if(arr[i] == NaN || arr[i] == undefined) print("Invalid input index " + i)
        
        let thought
        tf.tidy(()=>{
            thought = this.model.predict(tf.tensor2d([arr])).dataSync()
        })
        return thought
    }

    copy(){
        let modelCopy
        tf.tidy(()=>{
            modelCopy = new NeuralNetwork(this.layers_num, this.layers_act, this.layers_type)
            const weights  = this.model.getWeights()
            let clonedWeights = []
            for(let i = 0; i < weights.length; i++)
                clonedWeights.push(weights[i].clone())
            modelCopy.model.setWeights(clonedWeights)    
        })
        return modelCopy
    }

    //Implement crossbreed

    mutate(mutationRate){
        tf.tidy(()=>{   
            let mutatedWeights
            const weights = this.model.getWeights()
            mutatedWeights = []
            for(let i = 0; i < weights.length; i++){
                let tensor = weights[i]
                let shape = tensor.shape
                let values = tensor.dataSync().slice()
                for(let j = 0; j < values.length; j++){
                    if(random(1) < mutationRate)
                        values[j] = randomGaussian()
                }
                let newTensor = tf.tensor(values, shape)
                mutatedWeights.push(newTensor)
            }
            this.model.setWeights(mutatedWeights)
        })
        return this
    }

    serialize(){
        return JSON.stringify(this)
    }
    deserialize(json){
        if(typeof json == 'string'){
            json = JSON.parse(json)
        }
        //Implement thiss
    }
}

class Renderer{
    constructor(args){
        this.buffer1 = args.textBuffer ? args.textBuffer : null
        this.buffer2 = args.renderBuffer ? args.renderBuffer : null
        this.tBX = args.tBX ? args.tBX : 0
        this.tBY = args.tBY ? args.tBY : 0
        this.rBX = args.rBX && this.buffer2 ? width-args.renderBuffer.width : 0
        this.rBY = args.rBY ? args.rBY : 0
    }
    showText(newt){
        //Render the text in the first buffer
        const buffer = this.buffer1
        buffer.background(25)

        buffer.fill(255)
        const textSize = (buffer.width * buffer.height)/1300
        buffer.textSize(textSize)
        buffer.text("Generation : " + newt.generation, 0, textSize)

        tint(255, 100)
        image(buffer, this.tBX, this.tBY)
    }
    showNet(brain){
        //Render the network in the second buffer
        const buffer = this.buffer2
        const shape = brain.layers_num
        buffer.background(25)
        const weights = brain.model.getWeights()
        const interval = weights.length/2+2
        //draw the inputs
        buffer.fill(80, 180, 120)
        for(let i = 1; i < shape[0]+1; ++i)
            buffer.ellipse(buffer.width/interval, buffer.height/(shape[0]+2)*i,
                (buffer.height/(shape[0]+2)*0.85))
        buffer.fill(80, 120, 180)
        buffer.ellipse(buffer.width/interval, buffer.height/(shape[0]+2)*(shape[0]+1),
            (buffer.height/(shape[0]+2)*0.85))
        
        //draw the layers
        for(let i = 0; i < weights.length; i+=2){
            const layer = weights[i].dataSync()
            const bias = weights[i+1].dataSync()
            
            for(let k = 0; k < shape[i/2+1]; ++k){
                //Nodes and weights
                for(let j = 0; j < layer.length/shape[i/2+1]; ++j){
                    const w = layer[j*shape[i/2+1]+k]
                    buffer.strokeWeight(abs(w*buffer.width/200)+0.1)
                    if(w > 0) buffer.stroke(0, 80, 255*w*buffer.width/200)
                    if(w < 0) buffer.stroke(255*abs(w*buffer.width/200), 80, 0)
                    if(w == 0) buffer.stroke(100, 100, 100)
                    buffer.line(buffer.width/interval*(i/2+1), buffer.height/(layer.length/shape[i/2+1]+2)*(j+1),
                                buffer.width/interval*(i/2+2), buffer.height/(shape[i/2+1]+2)*(k+1))
                    buffer.stroke(0)
                    buffer.strokeWeight(1)
                }
                buffer.fill(120, 120, 120)
                if(i === weights.length-2) buffer.fill(180, 80, 120)
                buffer.ellipse(buffer.width/interval*(i/2+2), buffer.height/(shape[i/2+1]+2)*(k+1),
                    (buffer.height/(shape[i/2+1]+2))*0.85)
                //Biases and weigths
                const w = bias[k]
                buffer.strokeWeight(abs(w*buffer.width/200)+0.1)
                if(w > 0) buffer.stroke(0, 80, 255*w*buffer.width/200)
                if(w < 0) buffer.stroke(255*abs(w*buffer.width/200), 80, 0)
                if(w == 0) buffer.stroke(100, 100, 100)
                //Think it's f***ed up somewhere
                buffer.line(buffer.width/interval*(i/2+1), buffer.height/(shape[i/2]+1)*(shape[i/2]),
                            buffer.width/interval*(i/2+2), buffer.height/(shape[i/2+1]+2)*(k+1))
                buffer.stroke(0)
                buffer.strokeWeight(1)
                buffer.fill(80, 120, 180)
                if(i !== weights.length-2)
                buffer.ellipse(buffer.width/interval*(i/2+2), buffer.height/(shape[i/2+1]+2)*(shape[i/2+1]+1),
                            (buffer.height/(shape[i/2+1]+2))*0.85)
            }

        }
        tint(255, 100)
        image(buffer, this.rBX, this.rBY)
    }
}