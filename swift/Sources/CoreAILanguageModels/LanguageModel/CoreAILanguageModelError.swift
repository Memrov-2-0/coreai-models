// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Errors raised when a Core AI language-model bundle cannot preserve the
/// publisher's generation or prompt-template contract.
public enum CoreAILanguageModelError: Error, LocalizedError, Sendable, Equatable {
    case missingGenerationConfiguration(path: String)
    case invalidGenerationConfiguration(path: String, reason: String)
    case chatTemplateApplicationFailed(reason: String)

    public var errorDescription: String? {
        switch self {
        case .missingGenerationConfiguration(let path):
            "Missing required generation configuration at \(path)."
        case .invalidGenerationConfiguration(let path, let reason):
            "Invalid generation configuration at \(path): \(reason)"
        case .chatTemplateApplicationFailed(let reason):
            "The model chat template could not be applied: \(reason)"
        }
    }
}
