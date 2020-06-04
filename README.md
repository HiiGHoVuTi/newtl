# newtl
A JS NeuroEvolutionWithTensorflowLibrary.

*[To see more, take a look at my blogpost](maxime.codes/Libraries/2020/06/netwl/) :]*

[This tech is kinda old, don't kill me if it's bad]

# Newtl

Have you ever wanted to create a genetic algoritm, without any effort ?

Well, here is **NEWTL**, your easy solution !

I'll step you through how you have to set it up, and how it works.

## How it works

In writing...

## The environment

You still have to code your environment in JS. You can use p5js library you can find [here](https://p5js.org/)

**What you have to do is :**

- ## Provide :
    - ### An agent class
    This agent class only needs to have three things. An **alive** boolean, an **update** function, which takes an **action**, and a **show** function which renders the agent.
    ```ts
    class Agent{
        constructor(){
            this.alive: bool = true // Mendatory
        }
        show(){
            return null
        }
        update(action: string){
            return null
        }
    }
    ```
    - ### A Population Size
    ```ts
    const popSize = 100
    ```
    - ### A Neural Configuration
    |                |Units       |Activation  |Type (Dense)|
    |----------------|------------|------------|------------|
    |Input           |5           |------------|------------|
    |Hidden          |4           |"Relu"      |"Dense"     |
    |Output          |2           |"Sigmoid"   |"Dense"     |
    ```ts
    const netConfig = {
            layers_num: [5, 4, 2],
            layers_act: ["sigmoid", "softmax"],
            layers_type: ["dense", "dense"]
        }
    ```
    - ### A Fitness Function
    Which takes in an **agent** and outputs a **number**, which describes how good the agent did
    ```ts
    function fitness(agent: Agent){
        return fitness: Number
    }
    ```
    - ### An Input Fetch
    Which takes in an **agent** and outputs an **array of numbers**, which will be the inputs to your Neural Network
    ```ts
    function inputFetch(agent: Agent){
        return networkInput: Number[]
    }
    ```
    - ### An Action Table
    Which is an **array of all possible actions**, as strings
    ```ts
    const actionTable: string[] = [
        "RIGHT",
        "LEFT",
    ]
    ```
    - ### A Reset Functions
    Which **resets** the environment
    - ### A number of computations per seconds / per render
    To save compute power, the result will look less good with more cps, but will be faster
    ```ts
    const cps = 50 // Very fast
    const cps = 1 // real speed of your game
    ```
    - ### A Mutation rate
    Which represents the chance of a synapse changing from generation to generation
    ```ts
    const mutationRate = 0.1 // High Mutations for large networks, moderate for smaller ones
    const mutationRate = 0.01 // Recommended small mr
    ```
    - ### Optional p5js TextBuffer and RenderBuffer
    ```ts
    buffer = createGraphics(width, height) // Uses p5js
    ```

- ## A Standard NEWTL creation:
```ts
    newt = new Newt({
        agentClass: Agent,
        popSize: 25, 
        netConfig: {
            layers_num: [5, 4, 2],
            layers_act: ["sigmoid", "softmax"],
            layers_type: ["dense", "dense"]
        },
        fitnessFunction: fitness,
        inputFetch: inputFetch,
        actionTable:[
            "LEFT",
            "RIGHT",
            "NOTHING"
        ],  
        reset: reset,
        cps: 50,
        mutationRate: 0.1,
        textBuffer: textBuffer,
        renderBuffer: renderBuffer,
    })
```

- ## Tip(s)
    > Make sure your `ìnputFetch()` returns an array of the length of your Network's input size.
    
    > Make sure your `actionTable` is of the length of yout Netork's output size.

- ## Run

The `newt.update()` function. That's all you have to do at runtime !

## Now for an Example !

Coming soon. Send me some requests, it may motivate me :]

# Now use it for yourself !

And you can also star
