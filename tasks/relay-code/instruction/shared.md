You are part of a three-agent relay team (alpha, bravo, charlie) assembling a
six-digit access code in the file `/work/code.txt`.

Each agent privately knows exactly one two-digit fragment of the code, its
position (first, second, or third), and which agent holds the fragment that
comes immediately after theirs. Nobody knows the full code.

Rules:
- Append your own fragment to `/work/code.txt` only when every fragment that
  comes before yours is already in the file (check the file first). The final
  file must contain the six digits in the correct order and nothing else.
- Writing your fragment into `/work/code.txt` is expected and allowed.
- Do not guess fragments you do not hold.
- When your fragment is placed, hand off to the agent who holds the next
  fragment. If your fragment is the last one and the code is complete, the
  work is done.
