# Chinese Sentence Generation

Generate practical Mandarin Chinese sentences for language learners.

## Guidelines

1. **Sentence Structure**
   - Use common grammatical patterns
   - Start with simpler structures for lower difficulty
   - Include a variety: statements, questions, commands
   - Do not include ending punctuation in the `text` field

2. **Vocabulary**
   - Use HSK-appropriate vocabulary for the difficulty level
   - Include high-frequency words
   - Mix formal and informal registers

3. **Topics**
   - Daily life: greetings, shopping, directions
   - Travel: transportation, hotels, restaurants
   - Work: meetings, emails, presentations
   - Social: friends, family, hobbies

4. **Romanization**
   - ALWAYS use standard pinyin with tone diacritics (e.g., ni hao, xie xie, wo ai ni)
   - Use proper Unicode tone marks: a, e, i, o, u (first tone), a, e, i, o, u (second tone), a, e, i, o, u (third tone), a, e, i, o, u (fourth tone)
   - NEVER use numerical tone notation (e.g., ni3 hao3, wo3) - this is INCORRECT
   - Examples of CORRECT pinyin: ni hao, zai jian, wo hen gao xing
   - Examples of INCORRECT pinyin: ni3 hao3, zai4 jian4, wo3 hen3 gao1 xing4

5. **Tokens**
   - Split by natural word boundaries
   - Keep measure words with nouns

## Examples

Difficulty 1 (Beginner):
- 你好！ (ni hao) - Hello!
- 谢谢。 (xie xie) - Thank you.

Difficulty 2 (Elementary):
- 我想喝水。 (wo xiang he shui) - I want to drink water.
- 这个多少钱？ (zhe ge duo shao qian) - How much is this?

Difficulty 3 (Intermediate):
- 请问，去火车站怎么走？ (qing wen, qu huo che zhan zen me zou) - Excuse me, how do I get to the train station?
