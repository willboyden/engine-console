import { EcView } from '../components/base.js';
import { h, clear } from '../dom.js';
import { t } from '../i18n.js';
import { items, fmtPct } from '../format.js';
import { btn, card, field, select, emptyBox, errorBox, skeleton, badge } from '../components/ui.js';
import { toastError } from '../components/toast.js';
import { messageEl } from '../components/chat-core.js';

// Randomise which instance is sent as "A" so position bias cannot leak into votes (blind mode).
export function assignSides(a, b, rand = Math.random) { return rand() < 0.5 ? [a, b] : [b, a]; }

// The backend runs both completions itself (non-streaming, in parallel) and hides model names until the vote.
class EcArena extends EcView {
  setup() {
    this.blind = true;
    this.setupHost = h('div', {}, skeleton(2));
    this.stage = h('div');
    this.board = h('div', {}, skeleton(3));
    this.append(h('h1', t('nav.arena')), card(t('arena.new'), this.setupHost), this.stage, card(t('arena.leaderboard'), this.board));
    this.init(); this.loadBoard();
  }
  async init() {
    try { this.insts = items(await this.api.instances()).filter((i) => i.state === 'ready'); } catch (e) { clear(this.setupHost).append(errorBox(e, () => this.init())); return; }
    if (!this._alive) return;
    if (this.insts.length < 2) { clear(this.setupHost).append(emptyBox(t('arena.need_two'), t('arena.need_two_hint'), btn(t('nav.launch'), { onClick: () => { location.hash = '#/launch'; } }))); return; }
    this.a = this.insts[0].id; this.b = this.insts[1].id;
    const opts = this.insts.map((i) => ({ value: i.id, label: `${i.name || i.id} (${i.repo_id})` }));
    this.prompt = h('textarea', { rows: 3, placeholder: t('arena.prompt_ph'), 'aria-label': t('arena.prompt') });
    this.goBtn = btn(t('arena.go'), { kind: 'primary', icon: 'arena', onClick: () => this.start() });
    clear(this.setupHost).append(h('div', { class: 'stack-v' }, h('div', { class: 'row gap wrap' },
      field(t('arena.model_a'), select(opts, this.a, (v) => { this.a = v; })), field(t('arena.model_b'), select(opts, this.b, (v) => { this.b = v; })),
      h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: true, onChange: (e) => { this.blind = e.target.checked; } }), h('span', t('arena.blind')))),
      field(t('arena.prompt'), this.prompt), h('div', { class: 'row' }, this.goBtn)));
  }
  async start() {
    const text = this.prompt.value.trim();
    if (!text) return;
    if (this.a === this.b) { toastError(new Error(t('arena.same'))); return; }
    const [ia, ib] = this.blind ? assignSides(this.a, this.b) : [this.a, this.b];
    this.goBtn.disabled = true;
    clear(this.stage).append(h('div', { class: 'state', role: 'status', 'aria-live': 'polite' }, skeleton(2), h('span', t('arena.waiting'))));
    try {
      const m = await this.api.createMatch({ prompt: text, instance_a: ia, instance_b: ib, blind: this.blind, max_tokens: 1024 });
      if (!this._alive) return;
      this.match = m;
      const side = (i, resp) => { const me = messageEl({ role: 'assistant', label: m.blind && !m.winner ? t(i ? 'arena.label_b' : 'arena.label_a') : (i ? m.model_b : m.model_a) || '' }); me.flush({ text: resp || '' }); return me; };
      const ma = side(0, m.response_a), mb = side(1, m.response_b);
      this.voteHost = h('div', { class: 'row gap wrap', role: 'group', 'aria-label': t('arena.vote') });
      clear(this.stage).append(h('div', { class: 'grid two arena-cols' }, h('div', { class: 'card' }, h('div', { class: 'card-body' }, ma.el)), h('div', { class: 'card' }, h('div', { class: 'card-body' }, mb.el))), this.voteHost);
      this.voteHost.append(...[['a', 'arena.a_wins'], ['b', 'arena.b_wins'], ['tie', 'arena.tie']].map(([w, k]) => btn(t(k), { kind: w === 'tie' ? '' : 'primary', onClick: () => this.vote(w) })));
    } catch (e) { clear(this.stage); toastError(e); }
    this.goBtn.disabled = false;
  }
  async vote(winner) {
    try {
      const r = await this.api.vote(this.match.id, { winner });
      clear(this.voteHost).append(h('p', { class: 'note ok', role: 'status' }, t('arena.recorded')),
        h('p', {}, h('strong', t('arena.reveal')), ' ', badge(`A: ${r.model_a || '?'}`, 'info'), ' ', badge(`B: ${r.model_b || '?'}`, 'info')));
      this.loadBoard();
    } catch (e) { toastError(e); }
  }
  async loadBoard() {
    try {
      const rows = items(await this.api.leaderboard()); if (!this._alive) return;
      if (!rows.length) { clear(this.board).append(emptyBox(t('arena.no_board'), t('arena.no_board_hint'))); return; }
      clear(this.board).append(h('div', { class: 'tablewrap' }, h('table', h('thead', h('tr', ['arena.rank', 'arena.model', 'arena.elo', 'arena.matches', 'arena.winrate'].map((k) => h('th', { scope: 'col' }, t(k))))),
        h('tbody', rows.map((r, i) => h('tr', h('td', { class: 'num' }, i + 1), h('td', r.model || r.repo_id), h('td', { class: 'num' }, Math.round(r.rating ?? r.elo)),
          h('td', { class: 'num' }, r.games ?? r.matches ?? '–'), h('td', { class: 'num' }, (r.games ?? r.matches) ? fmtPct(((r.wins ?? 0) / (r.games ?? r.matches)) * 100) : '–')))))));
    } catch (e) { if (this._alive) clear(this.board).append(errorBox(e, () => this.loadBoard())); }
  }
}
customElements.define('ec-arena', EcArena);
