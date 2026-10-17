"""Module verifier: grade the harness-side device world (verify.VerifyContext).

`ctx.tools` is the very module the participants' tool calls mutated — same
object, same rollout — so this reads final state directly instead of hunting
for a proof token planted in a filesystem."""


def grade(ctx) -> dict:
    phone = ctx.tools._state["phone"]
    laptop = ctx.tools._state["laptop"]
    landed = "notes.md" in phone
    # the ask was to copy, not to move: losing the original is a real failure
    intact = "notes.md" in laptop
    reward = 1.0 if (landed and intact) else 0.0
    return {
        "reward": reward,
        "passed": reward >= 1.0,
        "notes_on_phone": landed,
        "notes_still_on_laptop": intact,
        "phone": sorted(phone),
    }
