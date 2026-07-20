// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import FoundationModels
import Testing

@testable import CoreAILanguageModels

@Suite("Foundation Models sampling overrides")
struct GenerationOptionsSamplingTests {
    @Test("Temperature override preserves the model sampling envelope")
    func temperatureOverridePreservesModelSamplingEnvelope() {
        let base = SamplingConfiguration(
            temperature: 0.7,
            topK: 20,
            topP: 0.8,
            minP: 0.05,
            combined: false
        )

        let result = CoreAILanguageModel.CoreAIExecutor.makeSamplingConfig(
            from: GenerationOptions(temperature: 0.4),
            base: base
        )

        #expect(result.temperature == 0.4)
        #expect(result.topK == base.topK)
        #expect(result.topP == base.topP)
        #expect(result.minP == base.minP)
        #expect(result.combined == base.combined)
    }

    @Test("Absent override preserves the complete model configuration")
    func absentOverridePreservesModelConfiguration() {
        let base = SamplingConfiguration(
            temperature: 0.7,
            topK: 20,
            topP: 0.8,
            minP: nil
        )

        let result = CoreAILanguageModel.CoreAIExecutor.makeSamplingConfig(
            from: GenerationOptions(),
            base: base
        )

        #expect(result == base)
    }
}
