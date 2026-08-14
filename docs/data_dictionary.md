# Data Dictionary — Cricket Shot Explorer

This is the design document for the synthetic dataset. For each of the six
shot types, it defines the typical value and rough spread of all six features.

The Python generator mirrors these values. If a value is changed here, the
corresponding value in `src/generate_data.py` must also be changed.

---

## Columns

| Column | Meaning | Unit | Example |
| `bat_speed` | Bat speed at impact | km/h | 95 |
| `impact_height` | Height of ball contact off the ground | cm | 35 |
| `ball_length` | How far from the batter the ball pitched | m | 3.2 |
| `ball_line` | Sideways position vs the stumps (+ = off side) | cm | 55 |
| `timing` | How well-timed the shot was | 0–1 | 0.78 |
| `front_foot` | Whether the batter stepped forward | true/false | true |
| `shot_type` | Target label — the shot played | text | `cover_drive` |

### Reading `ball_length`

Small values represent full deliveries pitched close to the batter.

Large values represent shorter deliveries.

### Reading `ball_line`

- `0` = approximately at the stumps
- Positive = off side
- Negative = leg side

---

## Shot Profiles

Each numerical cell contains:

**typical value (spread)**

The spread represents the standard deviation.

`front_foot` is represented as a probability.

| Shot Type | Bat Speed | Impact Height | Ball Length | Ball Line | Timing | Front Foot |
| `cover_drive` | 95 (12) | 35 (14) | 3.3 (0.6) | +52 (23) | 0.75 (0.12) | 87% |
| `straight_drive` | 92 (13) | 32 (13) | 3.3 (0.6) | +8 (21) | 0.77 (0.12) | 87% |
| `pull` | 102 (14) | 94 (23) | 7.1 (1.1) | 0 (30) | 0.70 (0.15) | 12% |
| `cut` | 96 (14) | 75 (21) | 6.8 (1.1) | +70 (26) | 0.72 (0.14) | 18% |
| `flick` | 83 (14) | 33 (15) | 3.8 (0.7) | −32 (23) | 0.74 (0.13) | 60% |
| `defensive` | 55 (16) | 37 (18) | 4.5 (1.4) | +14 (28) | 0.61 (0.17) | 56% |

---

## Reasoning

### Cover Drive

A full delivery outside off is driven through the covers, generally with
a front-foot movement and a full bat swing.

### Straight Drive

A straight drive has a similar profile to a cover drive on most features,
but the delivery line is closer to the stumps.

This intentional similarity makes it one of the hardest classes to distinguish.

### Pull

A pull is generally played against a short delivery. It has a high contact
point, relatively high bat speed, and is usually played from the back foot.

### Cut

A cut is generally played against a short and wide delivery outside off.
It shares some characteristics with the pull but differs in ball line and
impact height.

### Flick

A flick is generally played toward the leg side against a fuller delivery.
Its ball line is therefore shifted toward the negative/leg-side region.

### Defensive

A defensive shot is controlled rather than aggressively hit. Its most useful
separator in this synthetic dataset is relatively low bat speed.

---

## Deliberate Overlap

The dataset intentionally contains overlap between similar classes.

Important overlaps include:

1. `cover_drive` ↔ `straight_drive`
2. `straight_drive` ↔ `flick`
3. `defensive` ↔ `flick`

This is intentional.

The internship brief warns that approximately 99% accuracy would indicate that
the synthetic classes are too cleanly separated.

The final dataset produced approximately 83.5% test accuracy.

---

## Sampling Rules

- Numeric features are sampled from normal distributions.
- `front_foot` is sampled using a Bernoulli probability.
- A fixed random seed of `42` is used.
- Values are clipped to sensible ranges.
- Rows are shuffled before writing the CSV.

### Feature Bounds

| Feature | Minimum | Maximum |
| `bat_speed` | 20 | 150 |
| `impact_height` | 0 | 180 |
| `ball_length` | 0.5 | 11 |
| `ball_line` | −120 | 160 |
| `timing` | 0 | 1 |

---

## Known Limitation

Features are sampled independently within each shot class.

In real cricket, features would have relationships with one another.

For example, bat speed and timing may be correlated.

Therefore this dataset is useful for demonstrating an ML pipeline but does
not represent real cricket biomechanics.