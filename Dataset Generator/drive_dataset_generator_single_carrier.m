
function generate_quadriga_multiset()

% GENERATE_QUADRIGA_MULTISET

% Generates a COMPLETE suite of datasets for CSI Prediction:

% 1. Training Set (Mixed)

% 2. 5x Test Sets (Specific scenarios/speeds for robust evaluation)

%

% COMPATIBILITY:

% - QuaDRiGa 2.8.1-0

% - Python h5py (Saves as -v7.3)

% - 2x2 MIMO

clc; clear; close all;


% Global Settings

base_freq = 3.5e9;

samples_per_m = 30; % Nyquist safe for 3.5GHz


%% 1. DEFINE BATCHES

% =========================================================================

batches = struct();

b_idx = 1;


% --- BATCH 1: TRAINING SET (Mixed Scenarios, Standard Speeds) ---

batches(b_idx).name = 'train_mixed';

batches(b_idx).num_runs = 250; % Increase this for real training (e.g. 1000)

batches(b_idx).scenarios = {'3GPP_38.901_UMa_LOS', '3GPP_38.901_UMi_LOS', ...

'3GPP_38.901_UMa_NLOS', '3GPP_38.901_UMi_NLOS'};

batches(b_idx).v_range = [3, 60];

b_idx = b_idx + 1;

% --- BATCH 2: TEST SET (UMa LOS - Easy) ---

batches(b_idx).name = 'test_uma_los';

batches(b_idx).num_runs = 20;

batches(b_idx).scenarios = {'3GPP_38.901_UMa_LOS'};

batches(b_idx).v_range = [10, 60];

b_idx = b_idx + 1;

% --- BATCH 3: TEST SET (UMi NLOS - Hard/Rich Multipath) ---

batches(b_idx).name = 'test_umi_nlos';

batches(b_idx).num_runs = 20;

batches(b_idx).scenarios = {'3GPP_38.901_UMi_NLOS'};

batches(b_idx).v_range = [10, 60];

b_idx = b_idx + 1;

% --- BATCH 4: TEST SET (High Speed / Highway) ---

batches(b_idx).name = 'test_high_speed';

batches(b_idx).num_runs = 5;

batches(b_idx).scenarios = {'3GPP_38.901_UMa_LOS', '3GPP_38.901_UMa_NLOS'};

batches(b_idx).v_range = [80, 120]; % 80 to 120 km/h

b_idx = b_idx + 1;

% --- BATCH 5: TEST SET (Pedestrian / Low Speed) ---

batches(b_idx).name = 'test_pedestrian';

batches(b_idx).num_runs = 20;

batches(b_idx).scenarios = {'3GPP_38.901_UMi_NLOS', '3GPP_38.901_UMi_LOS'};

batches(b_idx).v_range = [3, 5]; % 3 to 5 km/h

b_idx = b_idx + 1;


% --- BATCH 6: TEST SET (General Hold-out) ---

batches(b_idx).name = 'test_general';

batches(b_idx).num_runs = 20;

batches(b_idx).scenarios = {'3GPP_38.901_UMa_LOS', '3GPP_38.901_UMi_NLOS'};

batches(b_idx).v_range = [10, 60];


%% 2. EXECUTE GENERATION

% =========================================================================

fprintf('Starting Batch Generation...\n');


for i = 1:length(batches)

cfg = batches(i);

cfg.center_freq = base_freq;

cfg.samples_per_m = samples_per_m;

cfg.track_len = 150; % 30 meters per run


fname = sprintf('dataset_%s.mat', cfg.name);


fprintf('\n------------------------------------------------\n');

fprintf('Generating Batch %d/%d: %s\n', i, length(batches), fname);

fprintf('Scenarios: %s ...\n', cfg.scenarios{1});

fprintf('Speed: [%d - %d] km/h\n', cfg.v_range(1), cfg.v_range(2));

fprintf('------------------------------------------------\n');


% Call the worker function

run_simulation_batch(cfg, fname);

end


fprintf('\nAll batches completed successfully.\n');

end

%% ----------------------------------------------------------------------------

% WORKER FUNCTION

% ----------------------------------------------------------------------------

function run_simulation_batch(config, filename)

% Storage

csi_dataset = cell(config.num_runs, 1);

speed_dataset = cell(config.num_runs, 1);

pdp_dataset = cell(config.num_runs, 1);

meta_dataset = cell(config.num_runs, 1);


% QuaDRiGa Setup

s = qd_simulation_parameters;

s.center_frequency = config.center_freq;

s.use_absolute_delays = 1;

for run_idx = 1:config.num_runs


if mod(run_idx, 5) == 0 || run_idx == 1

fprintf(' > Run %d / %d\n', run_idx, config.num_runs);

end


% 1. Select Scenario

scen_idx = randi(length(config.scenarios));

curr_scenario = config.scenarios{scen_idx};


% 2. 2x2 MIMO Layout

l = qd_layout(s);


base_ant = qd_arrayant('dipole');


tx_ant = base_ant.copy;

tx_ant.no_elements = 4;

tx_ant.element_position = [0, 0; -0.25, 0.25; 0, 0];


rx_ant = base_ant.copy;

rx_ant.no_elements = 4;

rx_ant.element_position = [0, 0; -0.25, 0.25; 0, 0];


l.tx_array = tx_ant;

l.rx_array = rx_ant;


if contains(curr_scenario, 'UMi')

l.tx_position = [0; 0; 10];

else

l.tx_position = [0; 0; 25];

end


% 3. Track & Speed Profile

t = qd_track('linear', config.track_len, 0);

t.name = sprintf('Track%04d', run_idx); % Safe name


r_start = 50 + rand * 150;

phi_start = rand * 2 * pi;

t.initial_position = [r_start * cos(phi_start); r_start * sin(phi_start); 1.5];


% Variable Speed

vmin_ms = config.v_range(1) / 3.6;

vmax_ms = config.v_range(2) / 3.6;

K = randi([4, 6]);


weights = rand(1, K); weights = weights / sum(weights);

seg_speeds = vmin_ms + (vmax_ms - vmin_ms) * rand(1, K);

seg_dists = config.track_len * weights;

seg_times = seg_dists ./ seg_speeds;


t_nodes = [0, cumsum(seg_times)];

d_nodes = [0, cumsum(seg_dists)];

t.movement_profile = [t_nodes; d_nodes];


% 4. Interpolate & Attach

t.interpolate_positions(config.samples_per_m);

l.rx_track = t;

l.set_scenario(curr_scenario);


% 5. Get Channels

cn = l.get_channels();


if numel(cn) > 1

H_tensor = cn(1).coeff;

else

H_tensor = cn(1).coeff;

end


% Verify 2x2

[n_rx, n_tx, ~, ~] = size(H_tensor);

if n_rx ~= 2 || n_tx ~= 2

warning('Run %d: Unexpected size [%d x %d]. Skipping.', run_idx, n_rx, n_tx);

continue;

end


% 6. Speed Vector Reconstruction

pos = t.positions;

diffs = diff(pos, 1, 2);

step_dists = sqrt(sum(diffs.^2, 1));

cum_dist_snaps = [0, cumsum(step_dists)];


prof_d = t.movement_profile(2,:);

prof_t = t.movement_profile(1,:);


if max(cum_dist_snaps) > max(prof_d)

prof_d(end) = max(cum_dist_snaps);

end


time_snaps = interp1(prof_d, prof_t, cum_dist_snaps, 'linear', 'extrap');

dt = diff(time_snaps);

dt(dt < 1e-9) = 1e-9;


inst_speed = step_dists ./ dt;

inst_speed = [inst_speed, inst_speed(end)];


% 7. Store

csi_dataset{run_idx} = H_tensor;

speed_dataset{run_idx} = inst_speed;

pdp_dataset{run_idx} = squeeze(sum(sum(sum(abs(H_tensor).^2, 1), 2), 3));


meta = struct();

meta.scenario = curr_scenario;

meta_dataset{run_idx} = meta;

end


% SAVE (HDF5 / -v7.3)

fprintf(' Saving %s ... ', filename);

save(filename, 'csi_dataset', 'speed_dataset', 'pdp_dataset', 'meta_dataset', 'config', '-v7.3');

fprintf('Done.\n');

end