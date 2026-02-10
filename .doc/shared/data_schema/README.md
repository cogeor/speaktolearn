# shared/data_schema/ - Data Contracts

## Purpose

Defines the shared data format between the Python generation tool and the Flutter app. Both systems must agree on this contract.

## Folder Structure

```
shared/
└── data_schema/
    ├── README.md                    # This file
    ├── sentences.schema.json        # JSON Schema for dataset
    └── examples/
        └── sentences.zh.example.json  # Example dataset
```

---

## Dataset JSON Schema

### `sentences.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://speaktolearn.app/schemas/sentences.schema.json",
  "title": "SpeakToLearn Dataset",
  "description": "Language learning text sequence dataset",
  "type": "object",
  "required": ["schema_version", "dataset_id", "language", "generated_at", "items"],
  "properties": {
    "schema_version": {
      "type": "string",
      "description": "Schema version for compatibility checking",
      "pattern": "^\\d+\\.\\d+\\.\\d+$",
      "examples": ["1.0.0"]
    },
    "dataset_id": {
      "type": "string",
      "description": "Unique identifier for this dataset",
      "pattern": "^[a-z0-9_]+$",
      "examples": ["mandarin_core_v1", "japanese_basic_v1"]
    },
    "language": {
      "type": "string",
      "description": "BCP-47 language code",
      "pattern": "^[a-z]{2}(-[A-Z]{2})?$",
      "examples": ["zh-CN", "ja-JP", "ko-KR"]
    },
    "generated_at": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp when dataset was generated"
    },
    "items": {
      "type": "array",
      "description": "List of text sequences",
      "items": { "$ref": "#/$defs/TextSequence" },
      "minItems": 1
    }
  },
  "$defs": {
    "TextSequence": {
      "type": "object",
      "required": ["id", "text"],
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier",
          "pattern": "^ts_\\d{6}$",
          "examples": ["ts_000001"]
        },
        "text": {
          "type": "string",
          "description": "Primary text in target language",
          "minLength": 1
        },
        "romanization": {
          "type": "string",
          "description": "Pronunciation guide (pinyin, romaji, etc.)",
          "examples": ["nǐ hǎo", "konnichiwa"]
        },
        "gloss": {
          "type": "object",
          "description": "Translations keyed by language code",
          "additionalProperties": { "type": "string" },
          "examples": [{"en": "Hello", "de": "Hallo"}]
        },
        "tokens": {
          "type": "array",
          "description": "Individual words/characters for highlighting",
          "items": { "type": "string" },
          "examples": [["你", "好"]]
        },
        "tags": {
          "type": "array",
          "description": "Categorization tags",
          "items": { "type": "string" },
          "examples": [["hsk1", "greeting"]]
        },
        "difficulty": {
          "type": "integer",
          "description": "Difficulty level (1=easiest)",
          "minimum": 1,
          "maximum": 5
        },
        "example_audio": {
          "$ref": "#/$defs/ExampleAudio"
        }
      }
    },
    "ExampleAudio": {
      "type": "object",
      "required": ["voices"],
      "properties": {
        "voices": {
          "type": "array",
          "description": "Available voice examples",
          "items": { "$ref": "#/$defs/Voice" }
        }
      }
    },
    "Voice": {
      "type": "object",
      "required": ["id", "uri"],
      "properties": {
        "id": {
          "type": "string",
          "description": "Voice identifier",
          "examples": ["f1", "m1"]
        },
        "label": {
          "type": "object",
          "description": "Human-readable label by language",
          "additionalProperties": { "type": "string" },
          "examples": [{"en": "Female", "zh": "女声"}]
        },
        "uri": {
          "type": "string",
          "description": "Audio file URI",
          "pattern": "^(assets|file|https?)://",
          "examples": ["assets://examples/female/ts_000001.opus"]
        },
        "duration_ms": {
          "type": "integer",
          "description": "Audio duration in milliseconds",
          "minimum": 0
        }
      }
    }
  }
}
```

---

## Example Dataset

### `examples/sentences.zh.example.json`

```json
{
  "schema_version": "1.0.0",
  "dataset_id": "mandarin_core_v1",
  "language": "zh-CN",
  "generated_at": "2026-02-06T10:00:00Z",
  "items": [
    {
      "id": "ts_000001",
      "text": "你好！",
      "romanization": "nǐ hǎo",
      "gloss": {
        "en": "Hello!",
        "de": "Hallo!"
      },
      "tokens": ["你", "好"],
      "tags": ["hsk1", "greeting"],
      "difficulty": 1,
      "example_audio": {
        "voices": [
          {
            "id": "f1",
            "label": {"en": "Female", "zh": "女声"},
            "uri": "assets://examples/female/ts_000001.opus",
            "duration_ms": 850
          },
          {
            "id": "m1",
            "label": {"en": "Male", "zh": "男声"},
            "uri": "assets://examples/male/ts_000001.opus",
            "duration_ms": 920
          }
        ]
      }
    },
    {
      "id": "ts_000002",
      "text": "我想喝水。",
      "romanization": "wǒ xiǎng hē shuǐ",
      "gloss": {
        "en": "I want to drink water."
      },
      "tokens": ["我", "想", "喝", "水"],
      "tags": ["hsk1", "daily", "wants"],
      "difficulty": 1,
      "example_audio": {
        "voices": [
          {
            "id": "f1",
            "label": {"en": "Female"},
            "uri": "assets://examples/female/ts_000002.opus",
            "duration_ms": 1200
          },
          {
            "id": "m1",
            "label": {"en": "Male"},
            "uri": "assets://examples/male/ts_000002.opus",
            "duration_ms": 1150
          }
        ]
      }
    },
    {
      "id": "ts_000003",
      "text": "请问，去火车站怎么走？",
      "romanization": "qǐng wèn, qù huǒ chē zhàn zěn me zǒu",
      "gloss": {
        "en": "Excuse me, how do I get to the train station?"
      },
      "tokens": ["请问", "去", "火车站", "怎么", "走"],
      "tags": ["hsk2", "travel", "directions"],
      "difficulty": 2,
      "example_audio": {
        "voices": [
          {
            "id": "f1",
            "label": {"en": "Female"},
            "uri": "assets://examples/female/ts_000003.opus",
            "duration_ms": 2100
          }
        ]
      }
    }
  ]
}
```

---

## Field Specifications

### ID Format

| Field | Pattern | Example |
|-------|---------|---------|
| TextSequence.id | `ts_NNNNNN` | `ts_000001` |
| Voice.id | Short identifier | `f1`, `m1`, `f2` |
| dataset_id | snake_case | `mandarin_core_v1` |

### Language Codes

| Language | BCP-47 | Notes |
|----------|--------|-------|
| Chinese (Simplified) | `zh-CN` | Mainland China |
| Chinese (Traditional) | `zh-TW` | Taiwan |
| Japanese | `ja-JP` | |
| Korean | `ko-KR` | |

### URI Schemes

| Scheme | Usage | Example |
|--------|-------|---------|
| `assets://` | Bundled in app | `assets://examples/female/ts_000001.opus` |
| `file://` | Local file | `file:///data/cache/ts_000001.opus` |
| `https://` | Remote CDN | `https://cdn.example.com/examples/female/ts_000001.opus` |

### Difficulty Levels

| Level | Description | Example Vocabulary |
|-------|-------------|-------------------|
| 1 | Beginner | HSK 1, JLPT N5 |
| 2 | Elementary | HSK 2, JLPT N4 |
| 3 | Intermediate | HSK 3-4, JLPT N3 |
| 4 | Upper Intermediate | HSK 5, JLPT N2 |
| 5 | Advanced | HSK 6, JLPT N1 |

---

## Validation

### Python Validation

```python
import json
import jsonschema
from pathlib import Path

def validate_dataset(json_path: Path, schema_path: Path) -> bool:
    """Validate a dataset against the schema."""
    with open(schema_path) as f:
        schema = json.load(f)

    with open(json_path) as f:
        data = json.load(f)

    try:
        jsonschema.validate(data, schema)
        return True
    except jsonschema.ValidationError as e:
        print(f"Validation error: {e.message}")
        return False
```

### Dart Validation

```dart
// Validation happens during JSON parsing
// If parsing succeeds, data is valid

try {
  final dataset = DatasetDto.fromJson(json);
  // Valid
} on FormatException catch (e) {
  // Invalid JSON
} on TypeError catch (e) {
  // Schema mismatch
}
```

---

## Compatibility

### Versioning Strategy

Schema version follows semver:
- **Major**: Breaking changes (e.g., renamed required field)
- **Minor**: New optional fields
- **Patch**: Documentation/description changes

### Migration Notes

| From | To | Changes |
|------|------|---------|
| - | 1.0.0 | Initial schema |

### Forward Compatibility

Both Python and Dart should:
- Ignore unknown fields
- Use defaults for missing optional fields
- Check `schema_version` before parsing

```dart
// Check version compatibility
final data = jsonDecode(jsonString);
final version = data['schema_version'] as String;
if (!version.startsWith('1.')) {
  throw UnsupportedError('Unsupported schema version: $version');
}
```

---

## File Locations

### Python (Generator)

```
tools/sentence_gen/
├── output/
│   ├── sentences.zh.json              # Generated dataset
│   └── examples/
│       ├── female/
│       │   └── ts_000001.opus
│       └── male/
│           └── ts_000001.opus
```

### Flutter (Consumer)

```
apps/mobile_flutter/
├── assets/
│   ├── datasets/
│   │   └── sentences.zh.json          # Copied from output
│   └── examples/
│       ├── female/
│       │   └── ts_000001.opus
│       └── male/
│           └── ts_000001.opus
```

### pubspec.yaml Assets

```yaml
flutter:
  assets:
    - assets/datasets/
    - assets/examples/female/
    - assets/examples/male/
```
