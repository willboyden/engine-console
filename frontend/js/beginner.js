// Beginner mode: a browser-local preference (default ON). It shows each help summary inline under its label, starts the
// parameter form at the basic level and makes "Launch" open the guided wizard. Storage may throw (private mode): safe wrappers.
import { local } from './store.js';

export const BEGINNER_KEY = 'ec.beginner';
let on = local.get(BEGINNER_KEY, '1') !== '0';
const subs = new Set();

export const isBeginner = () => on;
export function setBeginner(value) {
  on = !!value;
  local.set(BEGINNER_KEY, on ? '1' : '0');
  for (const fn of [...subs]) fn(on);
}
export const onBeginner = (fn) => { subs.add(fn); return () => subs.delete(fn); };
/** The parameter-form level a form starts at: basic for beginners, advanced otherwise (expert stays behind "show more"). */
export const defaultLevel = () => (on ? 'basic' : 'advanced');
