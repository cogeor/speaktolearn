import 'package:flutter/material.dart';

import '../../../../app/theme.dart';
import '../../../progress/domain/sentence_rating.dart';

/// A row of 4 rating buttons for self-reported difficulty.
///
/// Displays Hard, Almost, Good, Easy buttons in a row.
/// Each button animates (expands) on tap before triggering the callback.
class RatingNavButtons extends StatefulWidget {
  /// Callback when a rating is selected.
  final void Function(SentenceRating rating) onRate;

  const RatingNavButtons({super.key, required this.onRate});

  @override
  State<RatingNavButtons> createState() => _RatingNavButtonsState();
}

class _RatingNavButtonsState extends State<RatingNavButtons>
    with SingleTickerProviderStateMixin {
  int? _tappedIndex;

  final _buttonColors = [
    AppTheme.ratingHard,
    AppTheme.ratingAlmost,
    AppTheme.ratingGood,
    AppTheme.ratingEasy,
  ];

  final _buttonLabels = ['Hard', 'Almost', 'Good', 'Easy'];

  final _buttonRatings = [
    SentenceRating.hard,
    SentenceRating.almost,
    SentenceRating.good,
    SentenceRating.easy,
  ];

  late AnimationController _controller;
  late Animation<double> _widthFactorAnim;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 100),
      vsync: this,
    );

    _widthFactorAnim = Tween<double>(
      begin: 1.0,
      end: 1.2,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));

    _controller.addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        _controller.reverse();
      } else if (status == AnimationStatus.dismissed && _tappedIndex != null) {
        final rating = _buttonRatings[_tappedIndex!];
        widget.onRate(rating);
        setState(() => _tappedIndex = null);
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _onTap(int index) {
    if (_controller.isAnimating) return;
    setState(() => _tappedIndex = index);
    _controller.forward();
  }

  @override
  Widget build(BuildContext context) {
    const buttonHeight = 80.0;
    const spacing = 4.0;

    return SizedBox(
      height: buttonHeight,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final buttonWidth = (constraints.maxWidth - spacing * 3) / 4;

          return Stack(
            children: [
              // The static row of buttons
              Row(
                children: List.generate(4, (index) {
                  return Padding(
                    padding: EdgeInsets.only(right: index < 3 ? spacing : 0),
                    child: SizedBox(
                      width: buttonWidth,
                      height: buttonHeight,
                      child: TextButton(
                        style: TextButton.styleFrom(
                          backgroundColor: _buttonColors[index],
                          foregroundColor: Colors.black,
                          shape: const RoundedRectangleBorder(),
                        ),
                        onPressed: () => _onTap(index),
                        child: Text(
                          _buttonLabels[index],
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  );
                }),
              ),

              // The expanding overlay button, positioned correctly
              if (_tappedIndex != null)
                _buildExpandingButtonOverlay(
                  buttonWidth,
                  buttonHeight,
                  spacing,
                ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildExpandingButtonOverlay(
    double buttonWidth,
    double buttonHeight,
    double spacing,
  ) {
    return AnimatedBuilder(
      animation: _widthFactorAnim,
      builder: (context, child) {
        final expandedWidth = buttonWidth * _widthFactorAnim.value;

        // Calculate the original left position of the tapped button
        final originalLeft = _tappedIndex! * (buttonWidth + spacing);

        // To keep center fixed, offset left by half the increase in width
        final left = originalLeft - (expandedWidth - buttonWidth) / 2;

        return Positioned(
          left: left,
          top: 0,
          width: expandedWidth,
          height: buttonHeight,
          child: TextButton(
            style: TextButton.styleFrom(
              backgroundColor: _buttonColors[_tappedIndex!],
              foregroundColor: Colors.black,
              shape: const RoundedRectangleBorder(),
            ),
            onPressed: () {}, // Disabled during animation
            child: Text(
              _buttonLabels[_tappedIndex!],
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ),
        );
      },
    );
  }
}
