// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("ToolCallParser")
struct ToolCallParserTests {
    @Test("Plain chat does not create a tool parser")
    func parserIsDisabledWithoutEnabledTools() {
        let parser = ToolCallParser.whenToolsEnabled(
            openMarker: "<tool_call>",
            closeMarker: "</tool_call>",
            toolsAreEnabled: false
        )

        #expect(parser == nil)
    }

    @Test("Tool sessions still parse complete tool calls")
    func parserIsEnabledForToolSessions() {
        var parser = ToolCallParser.whenToolsEnabled(
            openMarker: "<tool_call>",
            closeMarker: "</tool_call>",
            toolsAreEnabled: true
        )
        let events = parser?.consume(
            #"<tool_call>{"name":"lookup","arguments":{"query":"Darian"}}</tool_call>"#
        ) ?? []

        #expect(events.count == 1)
        guard case .toolCall(_, let name, let arguments) = events[0] else {
            Issue.record("Expected a tool-call event")
            return
        }
        #expect(name == "lookup")
        #expect(arguments.contains(#""query":"Darian""#))
    }
}

#endif
