/// A discriminated union representing either success or failure.
sealed class Result<T, E> {
  const Result();

  /// Returns true if this is a successful result.
  bool get isSuccess => this is Success<T, E>;

  /// Returns true if this is a failure result.
  bool get isFailure => this is Failure<T, E>;

  /// Returns the success value or null.
  T? get valueOrNull => switch (this) {
    Success(:final value) => value,
    Failure() => null,
  };

  /// Returns the error or null.
  E? get errorOrNull => switch (this) {
    Success() => null,
    Failure(:final error) => error,
  };

  /// Maps the success value.
  Result<U, E> map<U>(U Function(T value) transform) => switch (this) {
    Success(:final value) => Success(transform(value)),
    Failure(:final error) => Failure(error),
  };

  /// Maps the error value.
  Result<T, F> mapError<F>(F Function(E error) transform) => switch (this) {
    Success(:final value) => Success(value),
    Failure(:final error) => Failure(transform(error)),
  };

  /// Chains another Result-returning operation.
  Result<U, E> flatMap<U>(Result<U, E> Function(T value) transform) =>
      switch (this) {
        Success(:final value) => transform(value),
        Failure(:final error) => Failure(error),
      };

  /// Executes the appropriate callback based on the result.
  R when<R>({
    required R Function(T value) success,
    required R Function(E error) failure,
  }) => switch (this) {
    Success(:final value) => success(value),
    Failure(:final error) => failure(error),
  };
}

/// Represents a successful result containing a value.
final class Success<T, E> extends Result<T, E> {
  final T value;
  const Success(this.value);

  @override
  bool operator ==(Object other) =>
      identical(this, other) || other is Success<T, E> && other.value == value;

  @override
  int get hashCode => value.hashCode;

  @override
  String toString() => 'Success($value)';
}

/// Represents a failed result containing an error.
final class Failure<T, E> extends Result<T, E> {
  final E error;
  const Failure(this.error);

  @override
  bool operator ==(Object other) =>
      identical(this, other) || other is Failure<T, E> && other.error == error;

  @override
  int get hashCode => error.hashCode;

  @override
  String toString() => 'Failure($error)';
}
