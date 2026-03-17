/// Metadata captured alongside a WAV recording for ML training and analysis.
///
/// A sidecar JSON file (`{textSequenceId}.json`) is written next to every
/// WAV file in the recordings directory. This value object is serialised with
/// [toJson] and deserialised with [fromJson].
class RecordingMetadata {
  const RecordingMetadata({
    required this.textSequenceId,
    required this.pinyin,
    required this.text,
    required this.hskLevel,
    required this.overallScore,
    this.characterScores,
    required this.scoringMethod,
    required this.recordingTimestamp,
    required this.deviceModel,
    required this.osVersion,
    required this.appVersion,
    required this.audioSampleRate,
    required this.audioDurationMs,
  });

  /// The TextSequence this recording belongs to.
  final String textSequenceId;

  /// Pinyin romanisation from [TextSequence.romanization].
  final String pinyin;

  /// Chinese character text from [TextSequence.text].
  final String text;

  /// HSK vocabulary level (1–6 or 7–9 depending on standard).
  final int hskLevel;

  /// Overall pronunciation score (0–100) from [Grade.overall].
  final int overallScore;

  /// Per-character scores (0.0–1.0) from [Grade.characterScores].
  final List<double>? characterScores;

  /// Scoring method identifier from [Grade.method], e.g. `'asr_cer_v1'`.
  final String scoringMethod;

  /// ISO 8601 timestamp of when the recording was made.
  final String recordingTimestamp;

  /// Human-readable device model string derived from [Platform.operatingSystem]
  /// and [Platform.operatingSystemVersion].
  final String deviceModel;

  /// OS version string from [Platform.operatingSystemVersion].
  final String osVersion;

  /// Application version string (from pubspec / build config).
  final String appVersion;

  /// Audio sample rate in Hz (always 16000 for this app).
  final int audioSampleRate;

  /// Recording duration in milliseconds.
  final int audioDurationMs;

  Map<String, dynamic> toJson() => {
    'textSequenceId': textSequenceId,
    'pinyin': pinyin,
    'text': text,
    'hskLevel': hskLevel,
    'overallScore': overallScore,
    if (characterScores != null) 'characterScores': characterScores,
    'scoringMethod': scoringMethod,
    'recordingTimestamp': recordingTimestamp,
    'deviceModel': deviceModel,
    'osVersion': osVersion,
    'appVersion': appVersion,
    'audioSampleRate': audioSampleRate,
    'audioDurationMs': audioDurationMs,
  };

  factory RecordingMetadata.fromJson(Map<String, dynamic> json) =>
      RecordingMetadata(
        textSequenceId: json['textSequenceId'] as String,
        pinyin: json['pinyin'] as String,
        text: json['text'] as String,
        hskLevel: json['hskLevel'] as int,
        overallScore: json['overallScore'] as int,
        characterScores: (json['characterScores'] as List<dynamic>?)
            ?.map((e) => (e as num).toDouble())
            .toList(),
        scoringMethod: json['scoringMethod'] as String,
        recordingTimestamp: json['recordingTimestamp'] as String,
        deviceModel: json['deviceModel'] as String,
        osVersion: json['osVersion'] as String,
        appVersion: json['appVersion'] as String,
        audioSampleRate: json['audioSampleRate'] as int,
        audioDurationMs: json['audioDurationMs'] as int,
      );
}
