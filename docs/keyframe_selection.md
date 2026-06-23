# KEY FRAME SELECTION STRATEGIES

## 1. Rule-based frame selection

This selection method base on TransNetV2 scene segments output and sampling keyframe uniformly based on its duration.

| Scene duration | How many keyframe | Keyframe sampling intervals (%duration) |
|---|---|---|
| `<= 3 second` | 1 | 100% |
| ` <= 10 second ` | 3 | 33.3% |
| `> 10 second` | 5 | 20% |

**Note:** The first keyframe of each will be sampled 1/2 interval from 'scene_start'
Example: 25s scene: start - 10% -- 30% -- 50% -- 70% -- 90% - end
