# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Create model bundles from exported .aimodel files."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

METADATA_VERSION = "0.2"


class GenerationConfigurationError(RuntimeError):
    """Raised when an LLM bundle lacks a usable generation configuration."""


def bundle_llm_asset(
    bundle_path: Path,
    hf_model_id: str,
    hf_config: Any,
    compression: str,
    name: str,
) -> None:
    """Add tokenizer and metadata.json (0.2 schema) to an LLM bundle.

    Expects ``{name}.aimodel`` to already exist inside bundle_path.
    """
    _write_generation_config(bundle_path / "generation_config.json", hf_model_id)
    _write_tokenizer(bundle_path / "tokenizer", hf_model_id)
    _write_metadata(bundle_path, hf_model_id, hf_config, compression, name)


def _write_tokenizer(dest: Path, hf_model_id: str) -> None:
    logger.info(f"Saving tokenizer from {hf_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    tokenizer.save_pretrained(str(dest))


def _write_generation_config(dest: Path, hf_model_id: str) -> None:
    """Preserve and validate a model's required generation configuration."""
    local_source = Path(hf_model_id) / "generation_config.json"
    try:
        source = (
            local_source
            if local_source.is_file()
            else Path(hf_hub_download(repo_id=hf_model_id, filename="generation_config.json"))
        )
    except (EntryNotFoundError, HfHubHTTPError, OSError) as error:
        raise GenerationConfigurationError(
            f"{hf_model_id} is missing required generation_config.json"
        ) from error

    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationConfigurationError(
            f"{hf_model_id} has an unreadable generation_config.json"
        ) from error

    if not isinstance(payload, dict) or not isinstance(payload.get("do_sample"), bool):
        raise GenerationConfigurationError(
            f"{hf_model_id} generation_config.json must contain boolean do_sample"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    logger.info("Preserved generation configuration at %s", dest)


def _write_metadata(
    bundle_path: Path,
    hf_model_id: str,
    hf_config: Any,
    compression: str,
    name: str,
) -> None:
    metadata: dict[str, Any] = {
        "metadata_version": METADATA_VERSION,
        "kind": "llm",
        "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {
            "tokenizer": hf_model_id,
            "vocab_size": getattr(hf_config, "vocab_size", None),
            "max_context_length": getattr(hf_config, "max_position_embeddings", None),
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "source": {
            "model_definition": "torch",
            "hf_model_id": hf_model_id,
        },
        "compression": compression if compression != "none" else None,
        "compilation": {
            "date": datetime.now().astimezone().isoformat(),
            "targets": [],
        },
    }
    metadata_path = bundle_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {metadata_path}")
