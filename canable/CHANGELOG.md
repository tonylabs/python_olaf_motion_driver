# CHANGELOG

## 2026-04-07

### Robstride: Fix `SET_ZERO_POSITION` not working in `zero_pos.py`

The Robstride `SET_ZERO_POSITION` command (communication type 6) was not actually setting the zero position. Two issues were found and fixed:

1. **Wrong response handler**: `receive_status_frame()` asserts the response must be `OPERATION_STATUS` (type 2) or `FAULT_REPORT` (type 21). The `SET_ZERO_POSITION` response returns a different frame type, causing the response to be misinterpreted. Fixed by using `receive()` to consume the raw response instead.

2. **Missing data byte**: The Robstride protocol requires `data[0] = 0x01` to confirm the zero position set. Without it, the motor ignores the command.

3. **Motor needs re-enable after zeroing**: The motor enters a disabled state after `SET_ZERO_POSITION`. Added `bus.enable()` call after the command to resume operation.
