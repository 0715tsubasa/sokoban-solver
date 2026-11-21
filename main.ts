// Sokoban solver for make.fun's generated levels.
//
// Strategy: the generator builds each level as a one-wide hallway with 1-4
// rooms hanging off it, each room joined to the hallway by a single door
// cell and holding 3-4 boxes with as many goals. A box pushed out of its
// room can never come back, and the level's push optimum is the sum of the
// rooms' optima. So: split the level at the hallway row (removing it
// disconnects each room, door cell included), solve every room
// push-optimally with A*
// over pushes (min-cost matching heuristic on precomputed push distances,
// dead-square pruning), and stitch the room solutions together with plain
// walks along the hallway. Every room but the last is solved with an
// "exit" constraint (the player must be able to walk back out), and the
// room that loses the most from that constraint is solved last.
//
// If the decomposition does not fit, a whole-level push A* is the fallback.
// Everything is deterministic: budgets count expansions, never the clock.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];
const INF = 1e9;

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goal: boolean[];
  boxes: number[]; // sorted cell indexes
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall: boolean[] = new Array(w * h).fill(true);
  const goal: boolean[] = new Array(w * h).fill(false);
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = false;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = true;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// ---------- generic helpers over a cell graph ----------

/** BFS walk distances/parents on cells where isFree(cell) holds. */
function bfsWalk(
  start: number,
  isFree: (c: number) => boolean,
  w: number,
  h: number,
): { dist: Int32Array; prev: Int32Array } {
  const dist = new Int32Array(w * h).fill(-1);
  const prev = new Int32Array(w * h).fill(-1);
  const q: number[] = [start];
  dist[start] = 0;
  for (let head = 0; head < q.length; head++) {
    const c = q[head];
    const cx = c % w;
    const cy = (c / w) | 0;
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (dist[n] !== -1 || !isFree(n)) continue;
      dist[n] = dist[c] + 1;
      prev[n] = c;
      q.push(n);
    }
  }
  return { dist, prev };
}

/** Move string for a BFS prev-chain from start to target (both reachable). */
function walkString(prev: Int32Array, start: number, target: number, w: number): string {
  const chars: string[] = [];
  let c = target;
  while (c !== start) {
    const p = prev[c];
    const d = c - p;
    chars.push(d === -w ? 'u' : d === w ? 'd' : d === -1 ? 'l' : 'r');
    c = p;
  }
  return chars.reverse().join('');
}

// ---------- push-distance tables ----------

/**
 * For one goal, the minimum pushes to bring a box from any cell to the goal,
 * ignoring other boxes. canStand(cell): the player could stand there;
 * canHold(cell): a box may sit there. BFS over the reversed push graph.
 */
function pushDistances(
  goalCell: number,
  canStand: (c: number) => boolean,
  canHold: (c: number) => boolean,
  w: number,
  h: number,
): Int32Array {
  const dist = new Int32Array(w * h).fill(INF);
  if (!canHold(goalCell)) return dist;
  dist[goalCell] = 0;
  const q: number[] = [goalCell];
  for (let head = 0; head < q.length; head++) {
    const y = q[head];
    const yx = y % w;
    const yy = (y / w) | 0;
    for (const [dx, dy] of DIRS) {
      // A push in direction (dx,dy) moved a box from x=y-(dx,dy) to y; the
      // player stood at y-2*(dx,dy).
      const xx = yx - dx;
      const xy = yy - dy;
      const sx = yx - 2 * dx;
      const sy = yy - 2 * dy;
      if (xx < 0 || xy < 0 || xx >= w || xy >= h) continue;
      if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
      const x = xy * w + xx;
      const s = sy * w + sx;
      if (dist[x] !== INF || !canHold(x) || !canStand(s)) continue;
      dist[x] = dist[y] + 1;
      q.push(x);
    }
  }
  return dist;
}

/** Min-cost matching of boxes to goals by brute-force permutations (n <= 8). */
function matchingCost(cost: number[][]): number {
  const n = cost.length;
  if (n === 0) return 0;
  const m = cost[0].length;
  if (m < n) return INF;
  const used = new Array(m).fill(false);
  let best = INF;
  const dfs = (b: number, acc: number) => {
    if (acc >= best) return;
    if (b === n) {
      best = acc;
      return;
    }
    for (let g = 0; g < m; g++) {
      if (used[g] || cost[b][g] >= INF) continue;
      used[g] = true;
      dfs(b + 1, acc + cost[b][g]);
      used[g] = false;
    }
  };
  dfs(0, 0);
  return best;
}

// ---------- push A* on a sub-board ----------

interface SubBoard {
  w: number;
  h: number;
  /** Player may stand on these cells. */
  stand: boolean[];
  /** A box may sit on these cells (subset of stand). */
  hold: boolean[];
  goals: number[];
  entry: number; // player-only cell just outside the room, or -1
}

interface SubSolution {
  moves: string;
  pushes: number;
  endPlayer: number;
}

/**
 * Push-optimal A* on a sub-board. If requireExit, a goal state counts only
 * when the player can still reach board.entry. Returns null on failure.
 */
function solveSub(
  board: SubBoard,
  startBoxes: number[],
  startPlayer: number,
  requireExit: boolean,
  maxExpansions: number,
): SubSolution | null {
  const { w, h } = board;
  const goals = board.goals;
  const goalSet = new Set(goals);
  const dists = goals.map((g) =>
    pushDistances(g, (c) => board.stand[c], (c) => board.hold[c], w, h),
  );
  const dead = new Uint8Array(w * h);
  for (let c = 0; c < w * h; c++) {
    if (!board.hold[c] || goalSet.has(c)) continue;
    if (dists.every((d) => d[c] >= INF)) dead[c] = 1;
  }
  if (startBoxes.some((b) => dead[b])) return null;

  const heuristic = (boxes: number[]): number => {
    const cost = boxes.map((b) => dists.map((d) => d[b]));
    return matchingCost(cost);
  };

  interface Node {
    boxes: number[];
    player: number; // canonical: smallest reachable cell
    playerCell: number; // an actual reachable cell (for path building)
    g: number;
    f: number;
    parent: Node | null;
    segment: string; // moves from parent to this node
  }

  const keyOf = (boxes: number[], rep: number) => boxes.join(',') + '|' + rep;

  const heap: Node[] = [];
  const pushHeap = (n: Node) => {
    heap.push(n);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p].f <= heap[i].f) break;
      [heap[p], heap[i]] = [heap[i], heap[p]];
      i = p;
    }
  };
  const popHeap = (): Node => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && heap[l].f < heap[m].f) m = l;
        if (r < heap.length && heap[r].f < heap[m].f) m = r;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i], heap[m]];
        i = m;
      }
    }
    return top;
  };

  const startBoxesSorted = [...startBoxes].sort((a, b) => a - b);
  const h0 = heuristic(startBoxesSorted);
  if (h0 >= INF) return null;
  const startNode: Node = {
    boxes: startBoxesSorted,
    player: startPlayer,
    playerCell: startPlayer,
    g: 0,
    f: h0,
    parent: null,
    segment: '',
  };
  pushHeap(startNode);
  const seen = new Map<string, number>();
  seen.set(keyOf(startBoxesSorted, startPlayer), 0);
  let expansions = 0;

  while (heap.length > 0) {
    const node = popHeap();
    if (node.g > (seen.get(keyOf(node.boxes, node.player)) ?? INF)) continue;
    const boxSet = new Set(node.boxes);
    const solved = node.boxes.every((b) => goalSet.has(b));
    if (solved) {
      // Reconstruct, then (optionally) walk out to the entry cell.
      const parts: string[] = [];
      let cur: Node | null = node;
      let endPlayer = node.playerCell;
      while (cur && cur.parent) {
        parts.push(cur.segment);
        cur = cur.parent;
      }
      parts.reverse();
      let moves = parts.join('');
      if (requireExit && board.entry >= 0) {
        const { dist, prev } = bfsWalk(
          node.playerCell,
          (c) => board.stand[c] && !boxSet.has(c),
          w,
          h,
        );
        if (dist[board.entry] < 0) {
          // Not acceptable as a finished state; keep searching. (Marking the
          // state seen with its g is fine: a later path here costs >=.)
        } else {
          moves += walkString(prev, node.playerCell, board.entry, w);
          endPlayer = board.entry;
          return { moves, pushes: node.g, endPlayer };
        }
      } else {
        return { moves, pushes: node.g, endPlayer };
      }
      if (requireExit) continue;
    }
    if (++expansions > maxExpansions) return null;

    // Player reachability in this state.
    const reach = bfsWalk(
      node.playerCell,
      (c) => board.stand[c] && !boxSet.has(c),
      w,
      h,
    );
    for (let bi = 0; bi < node.boxes.length; bi++) {
      const b = node.boxes[bi];
      const bx = b % w;
      const by = (b / w) | 0;
      for (const [dx, dy, ch] of DIRS) {
        const stand = (by - dy) * w + (bx - dx);
        const dest = (by + dy) * w + (bx + dx);
        if (stand < 0 || stand >= w * h || dest < 0 || dest >= w * h) continue;
        if (reach.dist[stand] < 0) continue;
        if (!board.hold[dest] || boxSet.has(dest) || dead[dest]) continue;
        const newBoxes = [...node.boxes];
        newBoxes[bi] = dest;
        newBoxes.sort((a, b2) => a - b2);
        const ng = node.g + 1;
        // Canonical player position after the push: the box's old cell.
        const after = bfsWalk(b, (c) => board.stand[c] && !newBoxes.includes(c), w, h);
        let rep = b;
        for (let c = 0; c < w * h; c++) if (after.dist[c] >= 0 && c < rep) rep = c;
        const key = keyOf(newBoxes, rep);
        if ((seen.get(key) ?? INF) <= ng) continue;
        const hh = heuristic(newBoxes);
        if (hh >= INF) continue;
        seen.set(key, ng);
        const walk = walkString(reach.prev, node.playerCell, stand, w);
        pushHeap({
          boxes: newBoxes,
          player: rep,
          playerCell: b,
          g: ng,
          f: ng + hh,
          parent: node,
          segment: walk + ch,
        });
      }
    }
  }
  return null;
}

// ---------- room decomposition ----------

interface Room {
  cells: number[]; // interior cells plus the door cell
  door: number;
  entry: number; // hallway cell just outside the door
  boxes: number[];
  goals: number[];
}

function findRooms(level: Level): Room[] | null {
  const { w, h, wall } = level;
  const n = w * h;
  const boxSet = new Set(level.boxes);
  const goalSet = new Set<number>();
  for (let c = 0; c < n; c++) if (level.goal[c]) goalSet.add(c);

  // The player starts on the hallway: a one-wide row of floor running the
  // board's width. Removing the hallway row disconnects every room (its
  // interior plus its door cell, which sits in the wall band next to the
  // hallway) from the rest of the board.
  const hy = (level.player / w) | 0;
  const hallway = new Set<number>();
  for (let x = 0; x < w; x++) {
    const c = hy * w + x;
    if (!wall[c]) hallway.add(c);
  }
  if (!hallway.has(level.player) || hallway.size < 3) return null;

  const compOf = new Int32Array(n).fill(-1);
  const comps: number[][] = [];
  for (let c = 0; c < n; c++) {
    if (wall[c] || hallway.has(c) || compOf[c] !== -1) continue;
    const cells: number[] = [];
    const q = [c];
    compOf[c] = comps.length;
    for (let hd = 0; hd < q.length; hd++) {
      const cur = q[hd];
      cells.push(cur);
      const cx = cur % w;
      const cy = (cur / w) | 0;
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const nn = ny * w + nx;
        if (wall[nn] || hallway.has(nn) || compOf[nn] !== -1) continue;
        compOf[nn] = comps.length;
        q.push(nn);
      }
    }
    comps.push(cells);
  }

  const rooms: Room[] = [];
  for (const cells of comps) {
    // The room's door: its cell adjacent to the hallway row. Exactly one.
    let door = -1;
    let entry = -1;
    for (const c of cells) {
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const [dx, dy] of DIRS) {
        const nn = (cy + dy) * w + (cx + dx);
        if (hallway.has(nn)) {
          if (door !== -1 && door !== c) return null; // two doors: not our shape
          door = c;
          entry = nn;
        }
      }
    }
    if (door < 0) return null;
    const boxes = cells.filter((c) => boxSet.has(c)).sort((a, b) => a - b);
    const goals = cells.filter((c) => goalSet.has(c));
    if (boxes.length < 3 || boxes.length > 4 || boxes.length !== goals.length) {
      return null;
    }
    rooms.push({ cells, door, entry, boxes, goals });
  }
  const covered = new Set<number>();
  for (const r of rooms) r.boxes.forEach((b) => covered.add(b));
  if (covered.size !== level.boxes.length) return null;
  const coveredGoals = new Set<number>();
  for (const r of rooms) r.goals.forEach((g) => coveredGoals.add(g));
  if (coveredGoals.size !== goalSet.size) return null;
  return rooms;
}

// ---------- simulation (for stitching and verification) ----------

function simulate(
  level: Level,
  moves: string,
): { boxes: number[]; player: number; ok: boolean } {
  const { w, h, wall } = level;
  const boxes = new Set(level.boxes);
  let player = level.player;
  let ok = true;
  for (const ch of moves) {
    const d = DIRS.find(([, , c]) => c === ch.toLowerCase());
    if (!d) continue;
    const [dx, dy] = d;
    const nx = (player % w) + dx;
    const ny = ((player / w) | 0) + dy;
    if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
      ok = false;
      break;
    }
    const target = ny * w + nx;
    if (wall[target]) {
      ok = false;
      break;
    }
    if (boxes.has(target)) {
      const bx = nx + dx;
      const by = ny + dy;
      if (bx < 0 || by < 0 || bx >= w || by >= h) {
        ok = false;
        break;
      }
      const beyond = by * w + bx;
      if (wall[beyond] || boxes.has(beyond)) {
        ok = false;
        break;
      }
      boxes.delete(target);
      boxes.add(beyond);
    }
    player = target;
  }
  return { boxes: [...boxes].sort((a, b) => a - b), player, ok };
}

// ---------- whole-level fallback ----------

function solveWhole(level: Level): string | null {
  const board: SubBoard = {
    w: level.w,
    h: level.h,
    stand: level.wall.map((x) => !x),
    hold: level.wall.map((x) => !x),
    goals: [],
    entry: -1,
  };
  for (let c = 0; c < level.w * level.h; c++) if (level.goal[c]) board.goals.push(c);
  // The generic matching heuristic enumerates permutations; with many boxes
  // that is too slow, so cap: use it only up to 8 boxes (rooms have <= 4;
  // the fallback is a safety net for odd shapes).
  if (level.boxes.length > 8) return null;
  const sol = solveSub(board, level.boxes, level.player, false, 3_000_000);
  return sol ? sol.moves : null;
}

// ---------- main solve ----------

function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = findRooms(level);
  if (rooms && rooms.length > 0) {
    // Solve every room both ways.
    interface RoomPlan {
      room: Room;
      withExit: SubSolution | null;
      noExit: SubSolution | null;
      board: SubBoard;
    }
    const plans: RoomPlan[] = [];
    let feasible = true;
    for (const room of rooms) {
      const stand = new Array(level.w * level.h).fill(false);
      const hold = new Array(level.w * level.h).fill(false);
      for (const c of room.cells) {
        stand[c] = true;
        hold[c] = true;
      }
      stand[room.entry] = true; // player may step out, boxes may not follow
      const board: SubBoard = {
        w: level.w,
        h: level.h,
        stand,
        hold,
        goals: room.goals,
        entry: room.entry,
      };
      const withExit = solveSub(board, room.boxes, room.entry, true, 5_000_000);
      const noExit = solveSub(board, room.boxes, room.entry, false, 5_000_000);
      if (!noExit) feasible = false;
      plans.push({ room, withExit, noExit, board });
    }
    if (feasible) {
      // Try each room as the last one, best gain first: the last room is
      // solved without the exit rule; the others use their exit-rule
      // solution when they have one. Stitching walks on the full board, so
      // a room solved without the exit rule still stitches whenever the
      // player can physically walk out of its final position.
      const gains = plans.map(
        (p) => (p.withExit ? p.withExit.pushes : INF) - (p.noExit ? p.noExit.pushes : INF),
      );
      const lastCandidates = plans
        .map((_, i) => i)
        .sort((a, b) => gains[b] - gains[a]);
      for (const lastIdx of lastCandidates) {
        const order = plans.map((_, i) => i).filter((i) => i !== lastIdx);
        order.push(lastIdx);
        let moves = '';
        let cur: Level = level;
        let stitched = true;
        for (const i of order) {
          const p = plans[i];
          const isLast = i === lastIdx;
          const sol = isLast ? p.noExit! : (p.withExit ?? p.noExit!);
          // Walk from the current player position to this room's entry.
          const boxSetNow = new Set(cur.boxes);
          const { dist, prev } = bfsWalk(
            cur.player,
            (c) => !cur.wall[c] && !boxSetNow.has(c),
            level.w,
            level.h,
          );
          if (dist[p.room.entry] < 0) {
            stitched = false;
            break;
          }
          const walk = walkString(prev, cur.player, p.room.entry, level.w);
          const segment = walk + sol.moves;
          moves += segment;
          const after = simulate(cur, segment);
          if (!after.ok) {
            stitched = false;
            break;
          }
          cur = { ...cur, boxes: after.boxes, player: after.player };
        }
        if (stitched) {
          const final = simulate(level, moves);
          const allOn = final.boxes.every((b) => level.goal[b]);
          if (final.ok && allOn) return moves;
        }
      }
    }
  }
  const whole = solveWhole(level);
  if (whole) {
    const final = simulate(level, whole);
    if (final.ok && final.boxes.every((b) => level.goal[b])) return whole;
  }
  return '';
}

if (process.env.MAKEFUN_NO_RUN !== '1') {
  for (const id of readdirSync(I).sort()) {
    const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
    mkdirSync(join(O, id), { recursive: true });
    writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
  }
}

export { parseLevel, findRooms, solveSub, solveLevel, simulate };
