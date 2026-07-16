// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Model-owned generation defaults preserved from a Hugging Face
/// `generation_config.json` file.
public struct LanguageGenerationConfiguration: Decodable, Sendable {
    public let doSample: Bool?
    public let temperature: Double?
    public let topK: Int?
    public let topP: Double?
    public let minP: Double?
    public let repetitionPenalty: Double?
    public let presencePenalty: Double?

    private enum CodingKeys: String, CodingKey {
        case doSample = "do_sample"
        case temperature
        case topK = "top_k"
        case topP = "top_p"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
    }

    /// Converts the supported model defaults into Core AI's native sampler.
    /// Unsupported values remain available on this type for future runtimes.
    public var samplingConfiguration: SamplingConfiguration {
        guard doSample != false else { return .greedy }

        let resolvedTemperature: Double
        if let temperature, temperature.isFinite, temperature >= 0 {
            resolvedTemperature = temperature
        } else if doSample == true {
            resolvedTemperature = 1
        } else {
            return .greedy
        }

        guard resolvedTemperature > 0 else { return .greedy }
        let resolvedTopK = topK.flatMap { $0 > 0 ? $0 : nil }
        let resolvedTopP = topP.flatMap { $0.isFinite && $0 > 0 && $0 <= 1 ? $0 : nil }
        let resolvedMinP = minP.flatMap { $0.isFinite && $0 > 0 && $0 <= 1 ? $0 : nil }

        return SamplingConfiguration(
            temperature: resolvedTemperature,
            topK: resolvedTopK,
            topP: resolvedTopP,
            minP: resolvedMinP
        )
    }
}
