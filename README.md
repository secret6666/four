# four
  func secondOrderUpdate(
        modelOutputs: [MLShapedArray<Float32>],
        timesteps: [Int],
        prevTimestep t: Int,
        sample: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        let (s0, s1) = (timesteps[back: 1], timesteps[back: 2])
        let (m0, m1) = (modelOutputs[back: 1], modelOutputs[back: 2])
        let (p_lambda_t, lambda_s0, lambda_s1) = (Double(lambda_t[t]), Double(lambda_t[s0]), Double(lambda_t[s1]))
        let p_alpha_t = Double(alpha_t[t])
        let (p_sigma_t, sigma_s0) = (Double(sigma_t[t]), Double(sigma_t[s0]))
        let (h, h_0) = (p_lambda_t - lambda_s0, lambda_s0 - lambda_s1)
        let r0 = h_0 / h
        let D0 = m0
        
        // D1 = (1.0 / r0) * (m0 - m1)
        let D1 = weightedSum(
            [1/r0, -1/r0],
            [m0, m1]
        )
        
        // See https://arxiv.org/abs/2211.01095 for detailed derivations
        // x_t = (
        //     (sigma_t / sigma_s0) * sample
        //     - (alpha_t * (torch.exp(-h) - 1.0)) * D0
        //     - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
        // )
        let x_t = weightedSum(
            [p_sigma_t/sigma_s0, -p_alpha_t * (exp(-h) - 1), -0.5 * p_alpha_t * (exp(-h) - 1)],
            [sample, D0, D1]
        )
        return x_t
    }

    public func step(output: MLShapedArray<Float32>, timeStep t: Int, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
        let prevTimestep = stepIndex == timeSteps.count - 1 ? 0 : timeSteps[stepIndex + 1]

        let lowerOrderFinal = useLowerOrderFinal && stepIndex == timeSteps.count - 1 && timeSteps.count < 15
        let lowerOrderSecond = useLowerOrderFinal && stepIndex == timeSteps.count - 2 && timeSteps.count < 15
        let lowerOrder = lowerOrderStepped < 1 || lowerOrderFinal || lowerOrderSecond
        
        let modelOutput = convertModelOutput(modelOutput: output, timestep: t, sample: sample)
        if modelOutputs.count == solverOrder { modelOutputs.removeFirst() }
        modelOutputs.append(modelOutput)
        
        let prevSample: MLShapedArray<Float32>
        if lowerOrder {
            prevSample = firstOrderUpdate(modelOutput: modelOutput, timestep: t, prevTimestep: prevTimestep, sample: sample)
        } else {
            prevSample = secondOrderUpdate(
                modelOutputs: modelOutputs,
                timesteps: [timeSteps[stepIndex - 1], t],
                prevTimestep: prevTimestep,
                sample: sample
            )
        }
        if lowerOrderStepped < solverOrder {
            lowerOrderStepped += 1
        }
        
        return prevSample
