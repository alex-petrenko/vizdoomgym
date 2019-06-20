import logging
import os
import re
from time import sleep

import cv2
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from vizdoom import *

log = logging.getLogger(__name__)

CONFIGS = [
    ['basic.cfg', 3],  # 0
    ['deadly_corridor.cfg', 7],  # 1
    ['defend_the_center.cfg', 3],  # 2
    ['defend_the_line.cfg', 3],  # 3
    ['health_gathering.cfg', 3],  # 4
    ['my_way_home.cfg', 5],  # 5
    ['predict_position.cfg', 3],  # 6
    ['take_cover.cfg', 2],  # 7
    ['deathmatch.cfg', 20],  # 8
    ['health_gathering_supreme.cfg', 3],  # 9
    ['my_way_home_sparse.cfg', 5],  # 10
    ['my_way_home_very_sparse.cfg', 5],  # 11
    ['my_way_home_goal.cfg', 7],  # 12
    ['sptm_battle_navigation.cfg', 7],  # 13

    ['textured_maze_easy.cfg', 7],  # 14
    ['textured_maze_very_sparse.cfg', 7],  # 15
    ['textured_maze.cfg', 7],  # 16
    ['textured_maze_multi_goal.cfg', 7],  # 17
    ['textured_maze_super_sparse.cfg', 7],  # 18
    ['textured_maze_large_no_goal.cfg', 7],  # 19
    ['my_way_home_multi_goal.cfg', 7],  # 20
    ['my_way_home_multi_goal_random.cfg', 7],  # 21
    ['my_way_home_no_goal.cfg', 7],  # 22
    ['my_way_home_no_goal_random.cfg', 7],  # 23

    ['textured_maze_super_sparse_v2.cfg', 7],  # 24
    ['textured_maze_no_goal_random.cfg', 7],  # 25
    ['textured_maze_random_goal_v2.cfg', 7],  # 26

    ['D3_battle.cfg', 9],  # 27
    ['D4_battle2.cfg', 9],  # 28
]


class VizdoomEnv(gym.Env):

    def __init__(self,
                 level,
                 coord_limits=None,
                 max_histogram_length=200,
                 show_automap=False,
                 skip_frames=1,
                 level_map='map01'):
        self.initialized = False

        # init game
        self.level = level
        self.show_automap = show_automap
        self.coord_limits = coord_limits
        self.skip_frames = skip_frames
        self.game = None
        self.state = None

        self.curr_seed = 0
        self.level_map = level_map

        self.screen_w, self.screen_h, self.channels = 640, 480, 3
        self.screen_resolution = ScreenResolution.RES_640X480
        self.calc_observation_space()

        self.action_space = spaces.Discrete(CONFIGS[self.level][1])

        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.config_path = os.path.join(scenarios_dir, CONFIGS[self.level][0])
        self.variable_indices = self._parse_variable_indices(self.config_path)

        self.viewer = None

        # Histogram to track positional coverage
        self.max_histogram_length = max_histogram_length
        self.current_histogram, self.previous_histogram = None, None
        if self.coord_limits:
            X = (self.coord_limits[2] - self.coord_limits[0])
            Y = (self.coord_limits[3] - self.coord_limits[1])
            if X > Y:
                len_x = self.max_histogram_length
                len_y = int((Y / X) * self.max_histogram_length)
            else:
                len_x = int((X / Y) * self.max_histogram_length)
                len_y = self.max_histogram_length
            self.current_histogram = np.zeros((len_x, len_y), dtype=np.int32)
            self.previous_histogram = np.zeros_like(self.current_histogram)

        self.seed()

    def calc_observation_space(self):
        self.observation_space = spaces.Box(0, 255, (self.screen_h, self.screen_w, self.channels), dtype=np.uint8)

    def _ensure_initialized(self, mode='algo'):
        if self.initialized:
            # Doom env already initialized!
            return

        self.game = DoomGame()

        self.game.load_config(self.config_path)
        self.game.set_screen_resolution(self.screen_resolution)
        # Setting an invalid level map will cause the game to freeze silently
        self.game.set_doom_map(self.level_map)
        self.game.set_seed(self.rng.random_integers(0, 2**32-1))

        if mode == 'algo':
            self.game.set_window_visible(False)
        elif mode == 'human':
            self.game.add_game_args('+freelook 1')
            self.game.set_window_visible(True)
            self.game.set_mode(Mode.SPECTATOR)
        else:
            raise Exception('Unsupported mode')

        if self.show_automap:
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS)
            self.game.set_automap_rotate(False)
            self.game.set_automap_render_textures(False)

            # self.game.add_game_args("+am_restorecolors")
            # self.game.add_game_args("+am_followplayer 1")
            background_color = 'ffffff'
            self.game.add_game_args("+viz_am_center 1")
            self.game.add_game_args("+am_backcolor " + background_color)
            self.game.add_game_args("+am_tswallcolor dddddd")
            # self.game.add_game_args("+am_showthingsprites 0")
            self.game.add_game_args("+am_yourcolor " + background_color)
            self.game.add_game_args("+am_cheat 0")
            self.game.add_game_args("+am_thingcolor 0000ff")  # player color
            self.game.add_game_args("+am_thingcolor_item 00ff00")
            # self.game.add_game_args("+am_thingcolor_citem 00ff00")

        self.game.init()

        self.initialized = True

    def _parse_variable_indices(self, config):
        with open(config, 'r') as config_file:
            lines = config_file.readlines()
        lines = [l.strip() for l in lines]

        variable_indices = {}

        for line in lines:
            if line.startswith('#'):
                continue  # comment

            variables_syntax = r'available_game_variables[\s]*=[\s]*\{(.*)\}'
            match = re.match(variables_syntax, line)
            if match is not None:
                variables_str = match.groups()[0]
                variables_str = variables_str.strip()
                variables = variables_str.split(' ')
                for i, variable in enumerate(variables):
                    variable_indices[variable] = i
                break

        return variable_indices

    def _start_episode(self):
        # # TODO: Why set the seed here? Game is already initialized.
        # if self.curr_seed > 0:
        #     self.game.set_seed(self.curr_seed)
        #     self.curr_seed = self.rng.random_integers(0, 2**32 - 1)
        self.game.new_episode()
        return

    def seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed, max_bytes=4)
        self.rng, _ = seeding.np_random(seed=self.curr_seed)
        return [self.curr_seed, self.rng]

    def _game_variables_dict(self, state):
        game_variables = state.game_variables
        variables = {}
        for variable, idx in self.variable_indices.items():
            variables[variable] = game_variables[idx]
        return variables

    def step(self, action):
        self._ensure_initialized()
        info = {'num_frames': self.skip_frames}

        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()

        reward = self.game.make_action(act, self.skip_frames)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
            game_variables = self._game_variables_dict(state)
            info.update(self.get_info(game_variables))
            self._update_histogram(info)
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)

        return observation, reward, done, info

    def reset(self, mode='algo'):
        self._ensure_initialized(mode)

        self._start_episode()
        self.state = self.game.get_state()
        img = self.state.screen_buffer

        # Swap current and previous histogram
        if self.current_histogram is not None and self.previous_histogram is not None:
            swap = self.current_histogram
            self.current_histogram = self.previous_histogram
            self.previous_histogram = swap
            self.current_histogram.fill(0)

        return np.transpose(img, (1, 2, 0))

    def render(self, mode='human'):
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            h, w = img.shape[:2]
            render_w = 640

            if w < render_w:
                render_h = int(640 * h / w)
                img = cv2.resize(img, (render_w, render_h))

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer(maxwidth=800)
            self.viewer.imshow(img)
        except AttributeError:
            pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def play_human_mode(self, num_episodes=3):
        for episode in range(num_episodes):
            self.reset('human')
            while not self.game.is_episode_finished():
                self.game.advance_action()
                state = self.game.get_state()
                total_reward = self.game.get_total_reward()

                if state is not None:
                    print('===============================')
                    print('Info: ', self.get_info())
                    print('State: #' + str(state.number))
                    print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
                    print('Reward: \t' + str(self.game.get_last_reward()))
                    print('Total Reward: \t' + str(total_reward))

                    if self.show_automap and state.automap_buffer is not None:
                        map_ = state.automap_buffer
                        map_ = np.swapaxes(map_, 0, 2)
                        map_ = np.swapaxes(map_, 0, 1)
                        cv2.imshow('ViZDoom Automap Buffer', map_)
                        cv2.waitKey(28)
                    else:
                        sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
            if self.show_automap:
                cv2.destroyAllWindows()

        sleep(1)
        print('===============================')
        print('Done')

    def get_info(self, variables=None):
        if variables is None:
            variables = self._game_variables_dict(self.game.get_state())

        info_dict = {'pos': self.get_positions(variables)}
        info_dict.update(variables)
        return info_dict

    def get_info_all(self, variables=None):
        if variables is None:
            variables = self._game_variables_dict(self.game.get_state())
        info = self.get_info(variables)
        if self.previous_histogram is not None:
            info['previous_histogram'] = self.previous_histogram
        return info

    def get_positions(self, variables):
        return self._get_positions(variables)

    def _get_positions(self, variables):
        have_coord_data = True
        required_vars = ['POSITION_X', 'POSITION_Y', 'ANGLE']
        for required_var in required_vars:
            if required_var not in variables:
                have_coord_data = False
                break

        x = y = a = np.nan
        if have_coord_data:
            x = variables['POSITION_X']
            y = variables['POSITION_Y']
            a = variables['ANGLE']

        return {'agent_x': x, 'agent_y': y, 'agent_a': a}

    def get_automap_buffer(self):
        if self.game.is_episode_finished():
            return None
        state = self.game.get_state()
        map_ = state.automap_buffer
        map_ = np.swapaxes(map_, 0, 2)
        map_ = np.swapaxes(map_, 0, 1)
        return map_

    def _update_histogram(self, info, eps=1e-8):
        if self.current_histogram is None:
            return
        agent_x, agent_y = info['pos']['agent_x'], info['pos']['agent_y']

        # Get agent coordinates normalized to [0, 1]
        dx = (agent_x - self.coord_limits[0]) / (self.coord_limits[2] - self.coord_limits[0])
        dy = (agent_y - self.coord_limits[1]) / (self.coord_limits[3] - self.coord_limits[1])

        # Rescale coordinates to histogram dimensions
        # Subtract eps to exclude upper bound of dx, dy
        dx = int((dx - eps) * self.current_histogram.shape[0])
        dy = int((dy - eps) * self.current_histogram.shape[1])

        self.current_histogram[dx, dy] += 1


class VizdoomTexturedMazeEasy(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(14, coord_limits=(0, 0, 1856, 1856), **kwargs)


class VizdoomTexturedMazeVerySparse(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(15, coord_limits=(0, 0, 1856, 1856), **kwargs)


class VizdoomTexturedMaze(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(16, coord_limits=(0, 0, 1856, 1856), **kwargs)


class VizdoomTexturedMazeMultiGoal(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(17, coord_limits=(0, 0, 1856, 1856), **kwargs)


class VizdoomTexturedMazeSuperSparse(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(18, coord_limits=(0, 0, 2336, 2368), **kwargs)


class VizdoomTexturedMazeLargeNoGoal(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(19, coord_limits=(0, 0, 2336, 2368), **kwargs)


class VizdoomMyWayHomeMultiGoal(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(20, coord_limits=(160, -704, 1120, 128), **kwargs)


class VizdoomMyWayHomeMultiGoalRandom(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(21, coord_limits=(160, -704, 1120, 128), **kwargs)


class VizdoomMyWayHomeNoGoal(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(22, coord_limits=(160, -704, 1120, 128), **kwargs)


class VizdoomMyWayHomeNoGoalRandom(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(23, coord_limits=(160, -704, 1120, 128), **kwargs)


class VizdoomTexturedMazeSuperSparseV2(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(24, coord_limits=(0, 0, 2336, 2368), **kwargs)


class VizdoomTexturedMazeRandomGoalV2(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(26, coord_limits=(0, 0, 2336, 2368), **kwargs)


class VizdoomBattle(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(27, coord_limits=(0, 0, 1856, 1856), **kwargs)


class VizdoomBattle2(VizdoomEnv):
    def __init__(self, **kwargs):
        super().__init__(28, coord_limits=(0, 0, 1920, 1920), **kwargs)

