/// Base class for use cases that return a Future.
///
/// Use cases encapsulate a single unit of business logic.
/// Each use case has one method [run] that takes an [Input] and returns an [Output].
abstract class FutureUseCase<Input, Output> {
  /// Executes the use case with the given [input].
  Future<Output> run(Input input);
}

/// Base class for use cases that return a Stream.
///
/// Use this for operations that emit multiple values over time.
abstract class StreamUseCase<Input, Output> {
  /// Executes the use case with the given [input].
  Stream<Output> run(Input input);
}

/// Marker class for use cases that don't require input parameters.
class NoParams {
  const NoParams();
}
