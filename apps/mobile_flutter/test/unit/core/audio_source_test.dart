import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/core/audio/audio_source.dart';

void main() {
  group('AudioSource', () {
    group('AssetAudioSource', () {
      test('can be instantiated', () {
        const source = AssetAudioSource('assets/audio/test.opus');
        expect(source, isA<AudioSource>());
        expect(source, isA<AssetAudioSource>());
      });

      test('holds asset path', () {
        const source = AssetAudioSource('assets/audio/test.opus');
        expect(source.assetPath, 'assets/audio/test.opus');
      });

      test('equality works correctly', () {
        const source1 = AssetAudioSource('assets/audio/test.opus');
        const source2 = AssetAudioSource('assets/audio/test.opus');
        const source3 = AssetAudioSource('assets/audio/other.opus');

        expect(source1, equals(source2));
        expect(source1, isNot(equals(source3)));
      });

      test('hashCode is consistent with equality', () {
        const source1 = AssetAudioSource('assets/audio/test.opus');
        const source2 = AssetAudioSource('assets/audio/test.opus');

        expect(source1.hashCode, equals(source2.hashCode));
      });

      test('toString returns readable representation', () {
        const source = AssetAudioSource('assets/audio/test.opus');
        expect(source.toString(), 'AssetAudioSource(assets/audio/test.opus)');
      });
    });

    group('FileAudioSource', () {
      test('can be instantiated', () {
        const source = FileAudioSource('/data/recordings/test.m4a');
        expect(source, isA<AudioSource>());
        expect(source, isA<FileAudioSource>());
      });

      test('holds file path', () {
        const source = FileAudioSource('/data/recordings/test.m4a');
        expect(source.filePath, '/data/recordings/test.m4a');
      });

      test('equality works correctly', () {
        const source1 = FileAudioSource('/data/recordings/test.m4a');
        const source2 = FileAudioSource('/data/recordings/test.m4a');
        const source3 = FileAudioSource('/data/recordings/other.m4a');

        expect(source1, equals(source2));
        expect(source1, isNot(equals(source3)));
      });

      test('hashCode is consistent with equality', () {
        const source1 = FileAudioSource('/data/recordings/test.m4a');
        const source2 = FileAudioSource('/data/recordings/test.m4a');

        expect(source1.hashCode, equals(source2.hashCode));
      });

      test('toString returns readable representation', () {
        const source = FileAudioSource('/data/recordings/test.m4a');
        expect(source.toString(), 'FileAudioSource(/data/recordings/test.m4a)');
      });
    });

    group('UrlAudioSource', () {
      test('can be instantiated', () {
        const source = UrlAudioSource('https://example.com/audio.opus');
        expect(source, isA<AudioSource>());
        expect(source, isA<UrlAudioSource>());
      });

      test('holds URL', () {
        const source = UrlAudioSource('https://example.com/audio.opus');
        expect(source.url, 'https://example.com/audio.opus');
      });

      test('equality works correctly', () {
        const source1 = UrlAudioSource('https://example.com/audio.opus');
        const source2 = UrlAudioSource('https://example.com/audio.opus');
        const source3 = UrlAudioSource('https://example.com/other.opus');

        expect(source1, equals(source2));
        expect(source1, isNot(equals(source3)));
      });

      test('hashCode is consistent with equality', () {
        const source1 = UrlAudioSource('https://example.com/audio.opus');
        const source2 = UrlAudioSource('https://example.com/audio.opus');

        expect(source1.hashCode, equals(source2.hashCode));
      });

      test('toString returns readable representation', () {
        const source = UrlAudioSource('https://example.com/audio.opus');
        expect(
            source.toString(), 'UrlAudioSource(https://example.com/audio.opus)');
      });
    });

    group('Type checking and pattern matching', () {
      test('sources are distinct types', () {
        const asset = AssetAudioSource('path');
        const file = FileAudioSource('path');
        const url = UrlAudioSource('path');

        expect(asset, isNot(equals(file)));
        expect(asset, isNot(equals(url)));
        expect(file, isNot(equals(url)));
      });

      test('switch/case pattern matching works correctly', () {
        String getSourceType(AudioSource source) {
          return switch (source) {
            AssetAudioSource(:final assetPath) => 'asset: $assetPath',
            FileAudioSource(:final filePath) => 'file: $filePath',
            UrlAudioSource(:final url) => 'url: $url',
          };
        }

        expect(
          getSourceType(const AssetAudioSource('assets/test.opus')),
          'asset: assets/test.opus',
        );
        expect(
          getSourceType(const FileAudioSource('/path/test.m4a')),
          'file: /path/test.m4a',
        );
        expect(
          getSourceType(const UrlAudioSource('https://example.com/audio.opus')),
          'url: https://example.com/audio.opus',
        );
      });

      test('is checks work correctly', () {
        const AudioSource asset = AssetAudioSource('test');
        const AudioSource file = FileAudioSource('test');
        const AudioSource url = UrlAudioSource('test');

        expect(asset is AssetAudioSource, isTrue);
        expect(asset is FileAudioSource, isFalse);
        expect(asset is UrlAudioSource, isFalse);

        expect(file is AssetAudioSource, isFalse);
        expect(file is FileAudioSource, isTrue);
        expect(file is UrlAudioSource, isFalse);

        expect(url is AssetAudioSource, isFalse);
        expect(url is FileAudioSource, isFalse);
        expect(url is UrlAudioSource, isTrue);
      });
    });
  });
}
