// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import FoundationModels
import Testing

@testable import CoreAILanguageModels

@Suite("Core AI context options")
struct ContextOptionsTests {
    private static let prompt: [Transcript.Entry] = [
        .prompt(
            Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: "Hello"))]
            )
        )
    ]

    @Test("Ordinary chat disables Qwen thinking")
    func ordinaryChatDisablesThinking() throws {
        let recorder = ChatTemplateContextRecorder()
        let tokenizer = MergingMockTokenizer(contextRecorder: recorder)

        _ = try CoreAILanguageModel.CoreAIExecutor.makeTokens(
            from: Self.prompt,
            using: tokenizer,
            contextOptions: ContextOptions()
        )

        #expect(recorder.enableThinking == false)
    }

    @Test("Native reasoning levels enable Qwen thinking")
    func reasoningLevelsEnableThinking() throws {
        let levels: [ContextOptions.ReasoningLevel] = [
            .light,
            .moderate,
            .deep,
            .custom("deliberate"),
        ]

        for level in levels {
            let recorder = ChatTemplateContextRecorder()
            let tokenizer = MergingMockTokenizer(contextRecorder: recorder)

            _ = try CoreAILanguageModel.CoreAIExecutor.makeTokens(
                from: Self.prompt,
                using: tokenizer,
                contextOptions: ContextOptions(reasoningLevel: level)
            )

            #expect(recorder.enableThinking == true)
        }
    }

    @Test("Chat template failure remains typed")
    func chatTemplateFailureRemainsTyped() {
        let tokenizer = MergingMockTokenizer(failChatTemplate: true)

        #expect(throws: CoreAILanguageModelError.self) {
            _ = try CoreAILanguageModel.CoreAIExecutor.makeTokens(
                from: Self.prompt,
                using: tokenizer
            )
        }
    }
}
