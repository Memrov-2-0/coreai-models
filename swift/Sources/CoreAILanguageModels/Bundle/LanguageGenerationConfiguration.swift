// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Model-owned generation defaults preserved from a Hugging Face
/// `generation_config.json` file.
public struct LanguageGenerationConfiguration: Decodable, Sendable {
    public enum ValidationError: Error, LocalizedError, Sendable, Equatable {
        case missingDoSample
        case invalidTemperature(Double?)
        case invalidTopK(Int)
        case invalidTopP(Double)
        case invalidMinP(Double)

        public var errorDescription: String? {
            switch self {
            case .missingDoSample:
                "do_sample is required"
            case .invalidTemperature(let value):
                "temperature must be finite and greater than zero when sampling; got \(String(describing: value))"
            case .invalidTopK(let value):
                "top_k must be greater than zero; got \(value)"
            case .invalidTopP(let value):
                "top_p must be finite and in (0, 1]; got \(value)"
            case .invalidMinP(let value):
                "min_p must be finite and in (0, 1]; got \(value)"
            }
        }
    }

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
    public func validatedSamplingConfiguration() throws -> SamplingConfiguration {
        guard let doSample else { throw ValidationError.missingDoSample }
        guard doSample else { return .greedy }
        guard let temperature, temperature.isFinite, temperature > 0 else {
            throw ValidationError.invalidTemperature(temperature)
        }
        if let topK, topK <= 0 { throw ValidationError.invalidTopK(topK) }
        if let topP, !topP.isFinite || topP <= 0 || topP > 1 {
            throw ValidationError.invalidTopP(topP)
        }
        if let minP, !minP.isFinite || minP <= 0 || minP > 1 {
            throw ValidationError.invalidMinP(minP)
        }

        return SamplingConfiguration(
            temperature: temperature,
            topK: topK,
            topP: topP,
            minP: minP
        )
    }
}
