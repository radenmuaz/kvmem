from kvmem.hmn import load_config, train
import copy, shutil

hp = load_config('kvmem/configs/hmn_tpu_sanity_w25.py')
hp = copy.deepcopy(hp)
hp['name'] = '_repro_orig_stage0_b64'
hp['curriculum'] = hp['curriculum'][:1]
hp['curriculum'][0]['B'] = 64
hp['curriculum'][0]['n_steps'] = 100
hp['curriculum'][0]['eval_every'] = 100
hp['log_every'] = 5
shutil.rmtree('logs/_repro_orig_stage0_b64', ignore_errors=True)
train(hp, log_base='logs', device_str='tpu')
