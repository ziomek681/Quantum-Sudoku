# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:46:02 2025

@author: izuni
"""

# quantum_sudoku_final.py
import random
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tkinter import messagebox

# -------------------------
#  SOLVER / GENERATOR (jak wcześniej, z dtype fixes)
# -------------------------

class SudokuGenerator:
    def __init__(self, grid):
        self.grid = np.array(grid, dtype=int)

    def solve(self):
        A_eq, b_eq, bounds = self.build_constraints()
        c = np.zeros(729, dtype=float)
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if result.success:
            x = result.x
            if np.all((x < 1e-6) | (x > 1 - 1e-6)):
                return self.extract_solution(x)
        return None

    def build_constraints(self):
        A_eq, b_eq = [], []
        bounds = [(0, 1) for _ in range(729)]
        # 1) each cell exact 1
        for i in range(9):
            for j in range(9):
                row = np.zeros(729, dtype=float)
                for k in range(9):
                    row[i * 81 + j * 9 + k] = 1.0
                A_eq.append(row); b_eq.append(1.0)
        # 2) digit once in row
        for i in range(9):
            for k in range(9):
                row = np.zeros(729, dtype=float)
                for j in range(9):
                    row[i * 81 + j * 9 + k] = 1.0
                A_eq.append(row); b_eq.append(1.0)
        # 3) digit once in col
        for j in range(9):
            for k in range(9):
                row = np.zeros(729, dtype=float)
                for i in range(9):
                    row[i * 81 + j * 9 + k] = 1.0
                A_eq.append(row); b_eq.append(1.0)
        # 4) digit once in box
        for br in range(3):
            for bc in range(3):
                for k in range(9):
                    row = np.zeros(729, dtype=float)
                    for di in range(3):
                        for dj in range(3):
                            r = br*3 + di; c = bc*3 + dj
                            row[r * 81 + c * 9 + k] = 1.0
                    A_eq.append(row); b_eq.append(1.0)
        # 5) prefilled
        for i in range(9):
            for j in range(9):
                v = int(self.grid[i,j])
                if v != 0:
                    row = np.zeros(729, dtype=float)
                    k = v-1
                    row[i * 81 + j * 9 + k] = 1.0
                    A_eq.append(row); b_eq.append(1.0)
        return np.array(A_eq, dtype=float), np.array(b_eq, dtype=float), bounds

    def extract_solution(self, x):
        sol = np.zeros((9,9), dtype=int)
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    if x[i*81 + j*9 + k] > 0.5:
                        sol[i,j] = k+1
                        break
        return sol

    @staticmethod
    def generate_filled_sudoku():
        grid = np.zeros((9, 9), dtype=int)
        nums = list(range(1, 10))
        def is_valid(num, r, c):
            if num in grid[r, :] or num in grid[:, c]:
                return False
            rs, cs = 3*(r//3), 3*(c//3)
            if num in grid[rs:rs+3, cs:cs+3]:
                return False
            return True
        def fill():
            for i in range(9):
                for j in range(9):
                    if grid[i, j] == 0:
                        random.shuffle(nums)
                        for num in nums:
                            if is_valid(num, i, j):
                                grid[i, j] = num
                                if fill(): return True
                                grid[i, j] = 0
                        return False
            return True
        fill()
        return grid

    # fast backtracking MRV (as before)
    def _count_solutions_collect(self, grid_in, stop_after=2, collect_up_to=1):
        try:
            grid = np.array(grid_in, dtype=int).tolist()
        except Exception:
            grid = [[int(v) for v in row] for row in grid_in]
        ALL_MASK = (1 << 10) - 2
        def box_index(r,c): return (r//3)*3 + (c//3)
        def cand_mask(r,c): return ALL_MASK & ~(row[r] | col[c] | box[box_index(r,c)])
        def bits_to_vals(mask):
            for v in range(1,10):
                if mask & (1<<v): yield v
        row = [0]*9; col = [0]*9; box = [0]*9; empty = []
        for r in range(9):
            for c in range(9):
                v = grid[r][c]
                if v == 0:
                    empty.append((r,c))
                else:
                    if not (1 <= v <= 9): return 0, []
                    bit = 1 << v
                    b = box_index(r,c)
                    if (row[r] & bit) or (col[c] & bit) or (box[b] & bit):
                        return 0, []
                    row[r] |= bit; col[c] |= bit; box[b] |= bit
        count = 0
        solutions = []
        def dfs():
            nonlocal count
            if count >= stop_after: return
            best_i = -1; best_mask = 0; best_pc = 10
            for i,(r,c) in enumerate(empty):
                if grid[r][c] != 0: continue
                m = cand_mask(r,c)
                pc = m.bit_count()
                if pc == 0: return
                if pc < best_pc:
                    best_pc = pc; best_mask = m; best_i = i
                    if pc == 1: break
            if best_i == -1:
                count += 1
                if len(solutions) < collect_up_to:
                    solutions.append([row_[:] for row_ in grid])
                return
            empty[-1], empty[best_i] = empty[best_i], empty[-1]
            r,c = empty[-1]
            for val in bits_to_vals(best_mask):
                bit = 1 << val
                b = box_index(r,c)
                grid[r][c] = val
                row[r] |= bit; col[c] |= bit; box[b] |= bit
                dfs()
                grid[r][c] = 0
                row[r] &= ~bit; col[c] &= ~bit; box[b] &= ~bit
                if count >= stop_after:
                    break
            empty[-1], empty[best_i] = empty[best_i], empty[-1]
        dfs()
        return count, solutions

    def unique_solution(self):
        try:
            grid_np = self.grid
            grid = [[int(grid_np[r,c]) for c in range(9)] for r in range(9)]
        except Exception:
            grid = [row[:] for row in self.grid]
        ALL_MASK = (1 << 10) - 2
        def box_index(r,c): return (r//3)*3 + (c//3)
        def cand_mask(r,c): return ALL_MASK & ~(row[r] | col[c] | box[box_index(r,c)])
        def bits_to_vals(mask):
            for v in range(1,10):
                if mask & (1<<v): yield v
        row = [0]*9; col = [0]*9; box = [0]*9
        empty = []
        for r in range(9):
            for c in range(9):
                v = grid[r][c]
                if v == 0:
                    empty.append((r,c))
                else:
                    if not (1 <= v <= 9): return -1
                    bit = 1 << v
                    b = box_index(r,c)
                    if (row[r] & bit) or (col[c] & bit) or (box[b] & bit): return -1
                    row[r] |= bit; col[c] |= bit; box[b] |= bit
        solutions_found = 0
        def dfs():
            nonlocal solutions_found
            if solutions_found >= 2: return
            best_i = None; best_mask = 0; best_pc = 10
            for i,(r,c) in enumerate(empty):
                if grid[r][c] != 0: continue
                m = cand_mask(r,c)
                pc = m.bit_count()
                if pc == 0: return
                if pc < best_pc:
                    best_pc = pc; best_mask = m; best_i = i
                    if pc == 1: break
            if best_i is None:
                solutions_found += 1
                return
            r,c = empty[best_i]
            for val in bits_to_vals(best_mask):
                bit = 1<<val
                b = box_index(r,c)
                grid[r][c] = val
                row[r] |= bit; col[c] |= bit; box[b] |= bit
                dfs()
                grid[r][c] = 0
                row[r] &= ~bit; col[c] &= ~bit; box[b] &= ~bit
                if solutions_found >= 2: return
        dfs()
        if solutions_found == 0: return -1
        if solutions_found == 1: return 2
        return -2

    def _ambiguous_indices_from_solutions(self, solutions):
        if not solutions: return []
        base = solutions[0]
        ambiguous = []
        for idx in range(81):
            r,c = divmod(idx, 9)
            v0 = base[r][c]
            for s in solutions[1:]:
                if s[r][c] != v0:
                    ambiguous.append(idx); break
        return ambiguous

    def _given_indices(self, puzzle):
        try:
            return [r*9 + c for r in range(9) for c in range(9) if int(puzzle[r,c]) != 0]
        except Exception:
            return [r*9 + c for r in range(9) for c in range(9) if puzzle[r][c] != 0]

    def find_layout_with_many_solutions(self,
                                       min_sol=10, max_sol=100,
                                       min_amb=4,  max_amb=10,
                                       max_removed=35,
                                       samples_per_iter=12,
                                       max_iters=300,
                                       prefer_target=25,
                                       verbose=True):
        try:
            work = [[int(self.grid[r,c]) for c in range(9)] for r in range(9)]
        except Exception:
            work = [row[:] for row in self.grid]
        cnt0, _ = self._count_solutions_collect(work, stop_after=1, collect_up_to=0)
        if cnt0 != 1 and verbose:
            print(f"[WARN] Startowa plansza ma {cnt0} rozwiązań (lepiej startować z jednoznacznej).")
        givens_all = set(self._given_indices(work))
        removed = set()
        def rc(idx): return (idx//9, idx%9)
        best_tuple = None
        for it in range(1, max_iters+1):
            if len(removed) >= max_removed:
                if verbose: print(f"[STOP] Osiągnięto limit usunięć {max_removed}"); break
            pool = list(givens_all - removed)
            if not pool:
                if verbose: print("[STOP] Brak kolejnych danych do usunięcia"); break
            random.shuffle(pool)
            chosen_eval = None
            for idx in pool[:max(samples_per_iter, 1)]:
                r,c = rc(idx)
                old = work[r][c]; work[r][c] = 0
                count, sols = self._count_solutions_collect(work, stop_after=max_sol, collect_up_to=max_sol)
                if count <= max_sol:
                    score = abs(count - prefer_target)
                    if chosen_eval is None or score < chosen_eval[0]:
                        chosen_eval = (score, count, idx, [row[:] for row in work], sols)
                work[r][c] = old
                if chosen_eval and (min_sol <= chosen_eval[1] <= max_sol):
                    break
            if not chosen_eval:
                if verbose: print(f"[{it}] Wszystkie próby > {max_sol}."); continue
            _, count, idx, _, sols = chosen_eval
            r,c = rc(idx); work[r][c] = 0; removed.add(idx)
            if verbose:
                print(f"[{it}] Usunięto r{r+1}c{c+1} -> count={count}, removed={len(removed)}/{max_removed}")
            if best_tuple is None or abs(count - prefer_target) < best_tuple[0]:
                best_tuple = (abs(count - prefer_target), count, [row[:] for row in work], sols)
            if min_sol <= count <= max_sol:
                ambiguous = self._ambiguous_indices_from_solutions(sols)
                if min_amb <= len(ambiguous) <= max_amb:
                    return [row[:] for row in work], sols, ambiguous
                else:
                    if verbose: print(f"   → niejednoznacznych = {len(ambiguous)} (wymagane {min_amb}..{max_amb}), szukam dalej…")
        if best_tuple:
            _, _, best_grid, sols = best_tuple
            ambiguous = self._ambiguous_indices_from_solutions(sols)
            return best_grid, sols, ambiguous
        return work, [], []

    @staticmethod
    def remove_digits_until_unique(grid, num_to_remove=45):
        tries_uni = 0
        while True:
            puzzle = np.copy(grid)
            cells = list(range(81))
            random.shuffle(cells)
            for _ in range(min(num_to_remove, 81)):
                cell = cells.pop()
                r,c = divmod(cell, 9)
                puzzle[r,c] = 0
            template = SudokuGenerator(puzzle)
            tries_uni += 1
            wynik = template.unique_solution()
            if tries_uni > 1000:
                return puzzle, [], []
            if wynik > 0:
                grid_multi, solutions, ambiguous = template.find_layout_with_many_solutions(
                    min_sol=70, max_sol=200, min_amb=15, max_amb=40,
                    max_removed=10, samples_per_iter=30, max_iters=300,
                    prefer_target=100, verbose=False
                )
                return np.array(grid_multi, dtype=int), solutions, ambiguous
            # else try again

# -------------------------
#  GUI (matplotlib) - FINAL with corrections
# -------------------------

class Sudoku:
    def __init__(self, puzzle, solution, multi_solutions=None, ambiguous_indices=None):
        self.my_init(puzzle, solution, multi_solutions, ambiguous_indices)

        # figure setup
        self.tot_fig = plt.figure(figsize=(6, 7.3))
        self.subfigs = self.tot_fig.subfigures(3, 1, height_ratios=(0.5, 6, 0.8))
        self.fig_menu = self.subfigs[0]; self.fig = self.subfigs[1]; self.fig_counter = self.subfigs[2]
        self.ax = self.fig.subplots()
        self.ax_counter = self.fig_counter.subplots(); self.ax_menu = self.fig_menu.subplots()

        # styling (your palette)
        self.fig_menu.patch.set_facecolor('#f2cbe5')
        self.fig_counter.patch.set_facecolor('#f2cbe5')
        self.fig.patch.set_facecolor('#f2cbe5')
        self.ax.set_facecolor('#f2daea')
        self.ax.tick_params(direction='in', labelsize=0)
        self.ax_counter.tick_params(length=0); self.ax_counter.set_facecolor('#f8eaf3')
        self.ax_menu.tick_params(length=0); self.ax_menu.set_facecolor('#f2cbe5')
        for spine in self.ax_menu.spines.values(): spine.set_edgecolor('#f2cbe5')

        # events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.fig_menu.canvas.mpl_connect('button_press_event', self.onclick_menu)

        # initial draw
        self.draw_buttons(); self.draw_grid(); self.draw_counters()
        self.fig.canvas.manager.set_window_title("Quantum Sudoku")

    def my_init(self, puzzle, solution, multi_solutions=None, ambiguous_indices=None):
        self.puzzle = np.array(puzzle, dtype=int)
        self.solution = np.array(solution, dtype=int)
        self.user_grid = self.puzzle.copy()
        self.selected_cell = None

        # givens locked
        self.given_mask = (self.puzzle != 0)

        # all possible solutions (list of lists) and current indices referencing them
        self.all_solutions = [ [row[:] for row in s] for s in (multi_solutions or []) ]
        self.current_solution_indices = list(range(len(self.all_solutions)))
        self.multi_mode = len(self.all_solutions) > 0

        # ambiguous indices = those that differ between remaining solutions
        self.ambiguous_set = set(ambiguous_indices or [])

        # measured_sources: idx -> set(original_solution_indices) that produced this measurement
        self.measured_sources = {}   # e.g. {idx: {sol_idx1, sol_idx2}}
        self.measured_indices = set()

        # locked cells (givens + measurements + correct entries)
        self.locked_cells = set((r,c) for r in range(9) for c in range(9) if self.given_mask[r,c])

        # wrong (red) entries set
        self.wrong_entries = set()

        # last correct highlight
        self.last_correct = None

        # candidates
        self.candidates = {}
        self.amb_overlays = {}
        self.cand_text = {}

        # create entangled groups (local groups) from ambiguous_set
        self.entangled_groups = []
        self.index_to_group = {}
        if self.multi_mode and self.ambiguous_set:
            self._build_entangled_groups()

        # insert 2..4 additional measurements ONLY from ambiguous_set and each from different solution
        if self.multi_mode and self.ambiguous_set and self.all_solutions:
            self._insert_additional_measurements(num_additional=min(4, max(2, len(self.ambiguous_set))))

        if self.multi_mode:
            self._recompute_candidates()

    def _build_entangled_groups(self):
        amb = list(self.ambiguous_set)
        random.shuffle(amb)
        groups = []
        i = 0
        while i < len(amb):
            size = random.choice([2,2,3])  # prefer 2 sometimes 3
            group = amb[i:i+size]
            if len(group) >= 2:
                groups.append(group)
            i += len(group)
        self.entangled_groups = groups
        self.index_to_group = {}
        for gi,g in enumerate(groups):
            for idx in g:
                self.index_to_group[idx] = gi

    def _insert_additional_measurements(self, num_additional=2):
        sols = list(range(len(self.all_solutions)))
        random.shuffle(sols)
        chosen = []
        while len(chosen) < num_additional and sols:
            chosen.append(sols.pop())
        while len(chosen) < num_additional:
            chosen.append(random.randrange(len(self.all_solutions)))
        candidate_idxs = [idx for idx in self.ambiguous_set if self.user_grid[idx//9, idx%9] == 0]
        random.shuffle(candidate_idxs)
        used = set()
        for sol_idx in chosen:
            possible = [idx for idx in candidate_idxs if idx not in used]
            if not possible: break
            idx = random.choice(possible)
            r,c = divmod(idx, 9)
            val = self.all_solutions[sol_idx][r][c]
            self.user_grid[r,c] = val
            self.measured_sources[idx] = {sol_idx}
            self.measured_indices.add(idx)
            self.locked_cells.add((r,c))
            used.add(idx)

    # UI menu handlers
    def onclick_menu(self, event):
        if event.inaxes != self.ax_menu: return
        if event.xdata is None or event.ydata is None: return
        col = event.xdata; row = event.ydata
        if 0.1 <= row < 0.9 and 2.0 <= col < 2.8:
            self.new_game()
        elif 0.1 <= row < 0.9 and 3.0 <= col < 3.9:
            self.exit_game()

    def new_game(self):
        filled = SudokuGenerator.generate_filled_sudoku()
        puzzle, multi_sols, ambiguous = SudokuGenerator.remove_digits_until_unique(filled, num_to_remove=45)
        self.my_init(puzzle, filled, multi_solutions=multi_sols, ambiguous_indices=ambiguous)
        self.draw_grid(); self.draw_counters(); self.draw_buttons()

    def exit_game(self):
        plt.close(self.tot_fig)

    def draw_buttons(self):
        self.ax_menu.clear()
        self.ax_menu.set_xlim(0, 4); self.ax_menu.set_ylim(0, 1)
        self.ax_menu.set_xticks(np.arange(0, 4, 1)); self.ax_menu.set_yticks(np.arange(0, 1, 1))
        self.ax_menu.set_xticklabels([]); self.ax_menu.set_yticklabels([])
        rect1 = Rectangle((2.0, 0.1), 0.8, 0.8, linewidth=2, edgecolor='#d093bc', facecolor='#f1c4e1')
        rect2 = Rectangle((3.0, 0.1), 0.9, 0.8, linewidth=2, edgecolor='#d093bc', facecolor='#f1c4e1')
        self.ax_menu.add_patch(rect1); self.ax_menu.add_patch(rect2)
        self.ax_menu.text(2.4, 0.45, 'new game', ha='center', va='center', fontsize=11, color="black")
        self.ax_menu.text(3.46, 0.45, 'exit', ha='center', va='center', fontsize=11, color="black")
        self.fig_menu.canvas.draw_idle()

    # click/select cell
    def onclick(self, event):
        if event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return
        col = int(event.xdata); row = 8 - int(event.ydata)
        if 0 <= row < 9 and 0 <= col < 9:
            if (row, col) in self.locked_cells:
                return
            self.last_correct = None
            self.selected_cell = (row, col)
            self.draw_grid(); self.draw_counters()

    # key handler (entry)
    def onkey(self, event):
        if self.selected_cell is None: return
        row, col = self.selected_cell

        # backspace: only allowed when not locked (givens/measurements/correct)
        if event.key == "backspace":
            if not self.given_mask[row, col] and (row, col) not in self.locked_cells:
                # If it was a wrong entry, simply remove it and do NOT alter global splat state
                if (row, col) in self.wrong_entries:
                    self.user_grid[row, col] = 0
                    self.wrong_entries.discard((row, col))
                    # recompute candidates for ambiguous fields (but do not filter/prune)
                    if self.multi_mode:
                        self._recompute_candidates()
                    self.draw_grid(); self.draw_counters()
                    return
                # else (regular removable user entry) — remove and update solution filtering
                self.user_grid[row, col] = 0
                if self.multi_mode:
                    self._filter_current_solutions_by_user_grid()
                    self._prune_measured_sources()
                    self._recompute_candidates()
                self.draw_grid(); self.draw_counters()
            return

        # numeric input
        try:
            val = int(event.key)
        except Exception:
            return
        if not (1 <= val <= 9): return
        # cannot overwrite givens or locked_cells
        if self.given_mask[row, col] or (row, col) in self.locked_cells:
            return

        #print(self.user_grid)
        #print(self.solution)
        # CLASSIC MODE:
        if not self.multi_mode:
            self.user_grid[row, col] = val
            if self.solution[row, col] == val:
                # correct -> lock and temporary green highlight
                self.locked_cells.add((row, col))
                self.last_correct = (row, col)
                self.wrong_entries.discard((row, col))
            else:
                # wrong -> stays red until user changes
                self.wrong_entries.add((row, col))
            self.draw_grid(); self.draw_counters(); self.win()
            return

        # MULTI MODE:
        # compute remaining solution indices and values in this cell across them
        remaining = list(self.current_solution_indices)
        values_in_cell = {self.all_solutions[si][row][col] for si in remaining}
        # if entered value not compatible -> mark as wrong (red) and do nothing else
        if val not in values_in_cell:
            self.user_grid[row, col] = val
            self.wrong_entries.add((row, col))
            self.draw_grid()
            return

        # if the value is the same across ALL remaining solutions -> treat as ordinary correct (no collapse)
        if len(values_in_cell) == 1:
            self.user_grid[row, col] = val
            self.locked_cells.add((row, col))
            self.last_correct = (row, col)
            # store measured_sources as the current set (so later pruning will be consistent)
            self.measured_sources[row*9 + col] = set(self.current_solution_indices)
            self.measured_indices.add(row*9 + col)
            self.wrong_entries.discard((row, col))
            # do not modify ambiguous_set or other fields
            self.draw_grid(); self.draw_counters()
            return

        # otherwise: value compatible with a subset => treat as measurement
        compatible_indices = [si for si in remaining if self.all_solutions[si][row][col] == val]
        if not compatible_indices:
            # should not happen because we checked earlier, but safe-guard
            self.user_grid[row, col] = val
            self.wrong_entries.add((row, col))
            self.draw_grid()
            return

        # Narrow remaining solutions to the compatible subset, then choose ONE concrete solution
        # to model the "measurement collapse" for entangled group.
        chosen_sol_idx = random.choice(compatible_indices)
        # set current solutions to those compatible with the user's pick (we keep all compatible)
        self.current_solution_indices = compatible_indices

        # lock player's cell as measurement (source set = compatible_indices)
        self.user_grid[row, col] = val
        idx = row*9 + col
        self.measured_sources[idx] = set(compatible_indices)
        self.measured_indices.add(idx)
        self.locked_cells.add((row, col))
        self.wrong_entries.discard((row, col))
        self.last_correct = (row, col)
        

        # Now apply chosen solution's values ONLY to the entangled group containing this index (if exists).
        if idx in self.index_to_group:
            gi = self.index_to_group[idx]
            group = self.entangled_groups[gi]
            chosen_sol = self.all_solutions[chosen_sol_idx]
            for other_idx in group:
                r,c = divmod(other_idx, 9)
                if (r,c) == (row, col): continue
                # if there was a measured insertion from other sources that is now incompatible -> remove it
                if other_idx in self.measured_sources:
                    # if chosen_sol doesn't match that measured value -> remove that measured insertion
                    if chosen_sol[r][c] != self.user_grid[r,c]:
                        self.user_grid[r,c] = 0
                        self.measured_sources.pop(other_idx, None)
                        self.measured_indices.discard(other_idx)
                        self.locked_cells.discard((r,c))
                        # do not touch other groups or ambiguous_set
                        continue
                # otherwise, if currently empty -> insert chosen_sol value and lock it as measurement from chosen_sol_idx
                if self.user_grid[r,c] == 0:
                    self.user_grid[r,c] = chosen_sol[r][c]
                    self.measured_sources[other_idx] = {chosen_sol_idx}
                    self.measured_indices.add(other_idx)
                    self.locked_cells.add((r,c))

        # after measurement narrow ambiguous_set according to the remaining solutions
        remaining_sols = [self.all_solutions[si] for si in self.current_solution_indices]
        self.ambiguous_set = set(self._ambiguous_indices_from_solutions(remaining_sols))

        # prune measured entries whose source sets got eliminated
        self._prune_measured_sources()

        # recompute candidates
        self._recompute_candidates()

        self.draw_grid();
        self.draw_counters()
        # if only one current solution left -> ambiguous_set becomes empty (deterministic)
        if len(self.current_solution_indices) == 1:
            self.ambiguous_set.clear()
            final_index = self.current_solution_indices[0]
            self.solution = np.array(self.all_solutions[final_index])  # pełne rozwiązanie 9x9
            self.multi_mode = 0


    # drawing
    def draw_grid(self):
        self.ax.clear()
        self.ax.set_xlim(0, 9); self.ax.set_ylim(0, 9)
        self.ax.set_xticks(np.arange(0, 9, 1)); self.ax.set_yticks(np.arange(0, 9, 1))
        self.ax.set_xticklabels([]); self.ax.set_yticklabels([])

        for i in range(10):
            lw = 3 if i % 3 == 0 else 1.5
            self.ax.plot([i, i], [0, 9], color='black', linewidth=lw)
            self.ax.plot([0, 9], [i, i], color='black', linewidth=lw)

        # highlight selected cell
        if self.selected_cell:
            r,c = self.selected_cell
            rect = Rectangle((c, 8-r), 1, 1, linewidth=3, edgecolor="#d093bc", facecolor="#d093bc", alpha=1)
            self.ax.add_patch(rect)

        # draw cells
        for i in range(9):
            for j in range(9):
                idx = i*9 + j
                v = self.user_grid[i,j]

                # wrong entry full-cell highlight
                if (i,j) in self.wrong_entries:
                    rec = Rectangle((j, 8 - i), 1, 1, linewidth=0, edgecolor=None, facecolor="#cb294f", alpha=1)
                    self.ax.add_patch(rec)
                # temporary green highlight for last correct
                elif self.last_correct == (i,j):
                    rec = Rectangle((j, 8 - i), 1, 1, linewidth=0, edgecolor=None, facecolor="#b6cb29", alpha=1)
                    self.ax.add_patch(rec)

                # text color
                if v != 0:
                    if self.given_mask[i,j] or idx in self.measured_indices:
                        txt_color = "black"
                    else:
                        txt_color = "black"
                    self.ax.text(j + 0.5, 8.5 - i, str(v), ha='center', va='center', fontsize=16, color=txt_color)

                # draw candidate digits for ambiguous indices only (small font in corner)
                #if idx in self.candidates and self.user_grid[i,j] == 0:
                #    s = ''.join(str(vv) for vv in sorted(self.candidates[idx]))
                #    self.ax.text(j + 0.05, 8 - i + 0.05, s, fontsize=7, alpha=0.8, ha='left', va='bottom')

        # draw splątanie symbol for ambiguous_set
        for idx in self.ambiguous_set:
            r,c = divmod(idx, 9)
            tx = c + 0.78; ty = 8 - r + 0.78
            self.ax.text(tx, ty, "✦", fontsize=10, alpha=0.9, ha='center', va='center', color='#6b2e8a')

        self.fig.canvas.draw_idle()

    def counter(self):
        grid_counter = np.zeros(9, dtype=int)
        for i in range(9):
            for j in range(9):
                v = self.user_grid[i,j]
                if v != 0:
                    if (not self.multi_mode and self.solution[i,j] == v) or self.multi_mode:
                        grid_counter[v-1] += 1
        return grid_counter

    def draw_counters(self):
        self.ax_counter.clear()
        self.ax_counter.set_xlim(0, 9); self.ax_counter.set_ylim(0, 2)
        self.ax_counter.set_xticks(np.arange(0, 9, 1)); self.ax_counter.set_yticks(np.arange(0, 2, 1))
        self.ax_counter.set_xticklabels([]); self.ax_counter.set_yticklabels([])

        for i in range(10):
            self.ax_counter.plot([i, i], [0, 2], color='black', linewidth=1)
        for j in range(3):
            self.ax_counter.plot([0, 9], [j, j], color='black', linewidth=1)

        for i in range(9):
            self.ax_counter.text(i+0.5, 1.4, str(i+1), ha='center', va='center', fontsize=16, color="black")

        grid_counter = self.counter()
        for i in range(9):
            self.ax_counter.text(i+0.5, 0.43, str(9-grid_counter[i]), ha='center', va='center', fontsize=12, color="gray")

        self.fig_counter.canvas.draw_idle()

    # multi helpers
    def _recompute_candidates(self):
        if not self.multi_mode: return
        self.candidates.clear()
        remaining = self._current_solutions()
        for idx in list(self.ambiguous_set):
            r,c = divmod(idx, 9)
            if self.user_grid[r,c] != 0: continue
            vals = set(s[r][c] for s in remaining)
            self.candidates[idx] = vals

        # autofill singletons (treat as measurement)
        progress = True
        while progress:
            progress = False
            singletons = [(idx, next(iter(vals))) for idx, vals in self.candidates.items() if len(vals) == 1]
            for idx, val in singletons:
                r,c = divmod(idx, 9)
                if self.given_mask[r,c]: continue
                if self.user_grid[r,c] == val: continue
                self.user_grid[r,c] = val
                # measured_sources is copy of current_solution_indices (they all agree)
                self.measured_sources[idx] = set(self.current_solution_indices)
                self.measured_indices.add(idx)
                self.locked_cells.add((r,c))
                # filter current_solution_indices to those consistent (should be same)
                self.current_solution_indices = [si for si in self.current_solution_indices if self.all_solutions[si][r][c] == val]
                progress = True
            if progress:
                remaining = self._current_solutions()
                self.candidates.clear()
                for idx in list(self.ambiguous_set):
                    r,c = divmod(idx, 9)
                    if self.user_grid[r,c] != 0: continue
                    vals = set(s[r][c] for s in remaining)
                    self.candidates[idx] = vals

    def _current_solutions(self):
        return [self.all_solutions[i] for i in self.current_solution_indices]

    def _filter_current_solutions_by_user_grid(self):
        if not self.multi_mode: return
        new_indices = []
        for si in self.current_solution_indices:
            sol = self.all_solutions[si]
            ok = True
            for r in range(9):
                for c in range(9):
                    v = self.user_grid[r,c]
                    if v != 0 and sol[r][c] != v:
                        ok = False; break
                if not ok: break
            if ok: new_indices.append(si)
        self.current_solution_indices = new_indices
        remaining = self._current_solutions()
        self.ambiguous_set = set(self._ambiguous_indices_from_solutions(remaining))

    def _prune_measured_sources(self):
        if not self.measured_sources:
            return
        cur_set = set(self.current_solution_indices)
        to_remove = []
        for idx, src_set in list(self.measured_sources.items()):
            if src_set & cur_set:
                self.measured_sources[idx] = src_set & cur_set
            else:
                # removed: wipe the measured insertion and unlock the cell
                r,c = divmod(idx, 9)
                self.user_grid[r,c] = 0
                self.measured_indices.discard(idx)
                self.locked_cells.discard((r,c))
                to_remove.append(idx)
        for idx in to_remove:
            self.measured_sources.pop(idx, None)

    def _ambiguous_indices_from_solutions(self, solutions):
        if not solutions: return []
        base = solutions[0]
        ambiguous = []
        for idx in range(81):
            r,c = divmod(idx, 9)
            v0 = base[r][c]
            for s in solutions[1:]:
                if s[r][c] != v0:
                    ambiguous.append(idx); break
        return ambiguous

    # win / play / instructions
    def win(self):
        if not self.multi_mode and np.array_equal(self.user_grid, self.solution):
            response = messagebox.askquestion('WIN!', 'Wanna play again?', icon='warning')
            if response == 'yes':
                self.new_game()
            else:
                plt.close(self.tot_fig)

    def play(self):
        self._show_instructions()
        self.draw_grid(); self.draw_counters(); self.draw_buttons()
        plt.show()

    def _show_instructions(self):
        txt = (
            "Quantum Sudoku rules:\n\n"
            "• At the start there is more that 1 solution to the Sudoku\n"
            "• Cells with different fillings in solutions are entangled\n"
            "  (marked by symbol ✦).\n"
            "• At the start there are added 2-4 entangled starting numbers,\n"
            "  each from other solution.\n"
            "• If you guess correct number in entangled cell,\n"
            "  it's treated as measurment: only solutions with that filling are left\n"
            "  and that may modify/unentangle other cells\n"
            "• Correct fillings are marked as yellow and blocked\n"
            "• Wrong fillings are marked as red to the time you correct them.\n"
        )
        try:
            messagebox.showinfo("Game rules", txt)
        except Exception:
            print(txt)

# -------------------------
#  START
# -------------------------

if __name__ == "__main__":
    filled_sudoku = SudokuGenerator.generate_filled_sudoku()
    puzzle, multi_solutions, ambiguous = SudokuGenerator.remove_digits_until_unique(filled_sudoku, num_to_remove=45)
    game = Sudoku(puzzle, filled_sudoku, multi_solutions=multi_solutions, ambiguous_indices=ambiguous)
    game.play()
