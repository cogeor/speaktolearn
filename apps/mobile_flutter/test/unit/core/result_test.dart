import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/core/result.dart';

void main() {
  group('Result', () {
    group('Success', () {
      test('isSuccess returns true', () {
        const result = Success<int, String>(42);
        expect(result.isSuccess, isTrue);
      });

      test('isFailure returns false', () {
        const result = Success<int, String>(42);
        expect(result.isFailure, isFalse);
      });

      test('valueOrNull returns the value', () {
        const result = Success<int, String>(42);
        expect(result.valueOrNull, 42);
      });

      test('errorOrNull returns null', () {
        const result = Success<int, String>(42);
        expect(result.errorOrNull, isNull);
      });

      test('when calls success callback', () {
        const result = Success<int, String>(42);
        final output = result.when(
          success: (v) => 'value: $v',
          failure: (e) => 'error: $e',
        );
        expect(output, 'value: 42');
      });

      test('equality works correctly', () {
        const result1 = Success<int, String>(42);
        const result2 = Success<int, String>(42);
        const result3 = Success<int, String>(99);

        expect(result1, equals(result2));
        expect(result1, isNot(equals(result3)));
      });

      test('toString returns expected format', () {
        const result = Success<int, String>(42);
        expect(result.toString(), 'Success(42)');
      });
    });

    group('Failure', () {
      test('isSuccess returns false', () {
        const result = Failure<int, String>('error');
        expect(result.isSuccess, isFalse);
      });

      test('isFailure returns true', () {
        const result = Failure<int, String>('error');
        expect(result.isFailure, isTrue);
      });

      test('valueOrNull returns null', () {
        const result = Failure<int, String>('error');
        expect(result.valueOrNull, isNull);
      });

      test('errorOrNull returns the error', () {
        const result = Failure<int, String>('error');
        expect(result.errorOrNull, 'error');
      });

      test('when calls failure callback', () {
        const result = Failure<int, String>('error');
        final output = result.when(
          success: (v) => 'value: $v',
          failure: (e) => 'error: $e',
        );
        expect(output, 'error: error');
      });

      test('equality works correctly', () {
        const result1 = Failure<int, String>('error');
        const result2 = Failure<int, String>('error');
        const result3 = Failure<int, String>('different');

        expect(result1, equals(result2));
        expect(result1, isNot(equals(result3)));
      });

      test('toString returns expected format', () {
        const result = Failure<int, String>('error');
        expect(result.toString(), 'Failure(error)');
      });
    });

    group('map', () {
      test('transforms success value', () {
        const result = Success<int, String>(10);
        final mapped = result.map((v) => v * 2);
        expect(mapped.valueOrNull, 20);
        expect(mapped.isSuccess, isTrue);
      });

      test('preserves failure', () {
        const result = Failure<int, String>('error');
        final mapped = result.map((v) => v * 2);
        expect(mapped.errorOrNull, 'error');
        expect(mapped.isFailure, isTrue);
      });

      test('can change value type', () {
        const result = Success<int, String>(10);
        final mapped = result.map((v) => v.toString());
        expect(mapped.valueOrNull, '10');
      });
    });

    group('mapError', () {
      test('transforms error value', () {
        const result = Failure<int, String>('error');
        final mapped = result.mapError((e) => e.length);
        expect(mapped.errorOrNull, 5);
        expect(mapped.isFailure, isTrue);
      });

      test('preserves success', () {
        const result = Success<int, String>(10);
        final mapped = result.mapError((e) => e.length);
        expect(mapped.valueOrNull, 10);
        expect(mapped.isSuccess, isTrue);
      });

      test('can change error type', () {
        const result = Failure<int, String>('error');
        final mapped = result.mapError((e) => Exception(e));
        expect(mapped.errorOrNull, isA<Exception>());
      });
    });

    group('flatMap', () {
      test('chains successful operations', () {
        const result = Success<int, String>(10);
        final chained = result.flatMap((v) => Success(v.toString()));
        expect(chained.valueOrNull, '10');
        expect(chained.isSuccess, isTrue);
      });

      test('propagates inner failure', () {
        const result = Success<int, String>(10);
        final chained = result.flatMap<String>(
          (v) => const Failure('inner error'),
        );
        expect(chained.errorOrNull, 'inner error');
        expect(chained.isFailure, isTrue);
      });

      test('short-circuits on outer failure', () {
        const result = Failure<int, String>('outer error');
        var called = false;
        final chained = result.flatMap((v) {
          called = true;
          return Success(v.toString());
        });
        expect(called, isFalse);
        expect(chained.errorOrNull, 'outer error');
      });

      test('chains multiple operations', () {
        const result = Success<int, String>(5);
        final chained = result
            .flatMap((v) => Success(v * 2))
            .flatMap((v) => Success(v + 1));
        expect(chained.valueOrNull, 11);
      });
    });

    group('pattern matching', () {
      test('exhaustive switch on Success', () {
        const Result<int, String> result = Success(42);
        final output = switch (result) {
          Success(:final value) => 'Got: $value',
          Failure(:final error) => 'Error: $error',
        };
        expect(output, 'Got: 42');
      });

      test('exhaustive switch on Failure', () {
        const Result<int, String> result = Failure('oops');
        final output = switch (result) {
          Success(:final value) => 'Got: $value',
          Failure(:final error) => 'Error: $error',
        };
        expect(output, 'Error: oops');
      });
    });
  });
}
