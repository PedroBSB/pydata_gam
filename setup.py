anim_width  = 1000
anim_height = int(anim_width / 1.77777777)
anim_res    = 200
anim_dev    = "png"
ANIM_WIDTH   = 1000
ANIM_HEIGHT  = int(ANIM_WIDTH / 1.77777777)
ANIM_RES     = 200   # (not used directly by Plotly; kept for parity)
ANIM_DEV     = "png" # (not used directly; static export relies on kaleido)

# Animation timing
FRAME_DURATION_MS = 100   # HTML: time each frame displays
TRANSITION_MS     = 200   # HTML: transition duration between frames

# >>> FIX: target total GIF length (seconds), not per-frame duration
TOTAL_GIF_SECONDS = 5.0   # make the whole GIF last ~5 seconds

# Spline / data params
N        = 500    # number of x points
K        = 10     # number of basis functions (like k in R)
SPL_ORDER= 3      # cubic regression spline
N_STEPS   = 60