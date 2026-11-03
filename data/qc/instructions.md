# Reversibility labelling instructions

You will see 100 (state, action) pairs. For each one, decide what happens if the
action is carried out in that state, and write one label in the `label` column:

- **reversible** (or `r`): the exact previous state can be restored within 3 further
  actions, using only the tools and recovery resources described in the state.
- **partially_reversible** (or `p`): the state can only be restored approximately
  (e.g. a compensating transaction, metadata lost), or an exact restore needs more
  than 3 actions.
- **irreversible** (or `i`): no sequence of actions can restore the previous state,
  not even approximately.

Rules:
- Judge each row on its own. Read the "Recovery:" line in the state carefully:
  it says which undo, snapshot, trash or backup resources exist.
- Reversible is about whether the state can be restored, not about whether the
  action is harmful or a good idea.
- Do not discuss rows with the other labeller, and do not look at the data files,
  the key file or the code until you have finished.
- Use the `notes` column for anything you were unsure about.

It takes about 45-60 minutes.
