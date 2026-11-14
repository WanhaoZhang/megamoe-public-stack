const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const cp = require('child_process');
const OUT = __dirname;
const ROOT = process.env.MEGAMOE_SOURCE_ROOT && path.resolve(process.env.MEGAMOE_SOURCE_ROOT);
const NATIVE = ROOT ? path.join(ROOT, 'ascend_deepep_hkq') : path.resolve(OUT, '../..');
const REFERENCE = path.join(OUT, 'reference-src');
const EXP = path.resolve(process.env.MEGAMOE_EXPERIMENT_ROOT || path.join(REFERENCE, 'experiment'));
const A = 'ascend_deepep_hkq/';
const M = 'MindSpeed-LLM/';
const F = 'FSDPTurbo/';
const files = {
 runner:A+'integration/wuhan/run_training_case.sh', bootstrap:A+'integration/wuhan/runtime_bootstrap.py',
 train:M+'train_fsdp2.py', factory:M+'mindspeed_llm/fsdp2/models/model_factory.py',
 engine:M+'mindspeed_llm/fsdp2/distributed/mindspeed_parallel_engine.py',
 ep:F+'fsdp_turbo/distributed/expert_parallel/expert_parallel.py',
 efsdp:F+'fsdp_turbo/distributed/expert_parallel/expert_fully_shard_parallel.py',
 model:M+'mindspeed_llm/fsdp2/models/deepseek_v4/modeling_deepseek_v4.py',
 trainer:M+'mindspeed_llm/fsdp2/train/trainer.py',
 dispatcher:F+'fsdp_turbo/distributed/expert_parallel/dispatcher_megamoe_balanced.py',
 runtimeDispatcher:F+'fsdp_turbo/distributed/expert_parallel/dispatcher_mega_moe.py',
 utils:F+'fsdp_turbo/distributed/expert_parallel/utils.py',
 bridge:A+'ascend_deepep/fsdp2_bridge.py', device:A+'ascend_deepep/megaflux_device.py',
 worker:A+'ascend_deepep/megaflux_worker_pipeline.py', full:A+'ascend_deepep/megaflux_full.py',
 early:A+'ascend_deepep/megaflux_early.py', mixed:A+'ascend_deepep/megaflux_mixed.py',
 planner:A+'ascend_deepep/moonep_planner.py', validator:A+'ascend_deepep/execution_plan_device.py',
 workspace:A+'csrc/ops/mega_moe_bf16_replica.cpp', planKernel:A+'kernels/mega_moe_bf16/mega_moe_moonep_planner.cpp',
 torch:A+'csrc/torch_classes.cpp', host:A+'csrc/ops/mega_moe_bf16.cpp', launch:A+'csrc/launcher.cpp',
 kworker:A+'kernels/mega_moe_bf16/megaflux_worker_bf16.cpp',
 kfull:A+'kernels/mega_moe_bf16/megaflux_full_bf16.cpp',
 kh:A+'kernels/mega_moe_bf16/include/mega_moe_bf16.h',
 kbworker:A+'kernels/mega_moe_bf16/megaflux_worker_bf16_backward.cpp',
 kbfull:A+'kernels/mega_moe_bf16/megaflux_full_bf16_backward.cpp',
 kbh:A+'kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h',
 grad:A+'kernels/mega_moe_bf16/include/mega_moe_gradient_pipe.h',
 build:A+'scripts/build_megaflux_worker_pipeline.sh', cmake:A+'CMakeLists.txt',
 stage:'experiment/tools/whole-stage-a1.sh', run:'experiment/tools/run-whole-a1.sh',
 entry:'experiment/tools/whole_entry.py', fixed:'experiment/tools/fixed_python_route.py',
 mesh:F+'fsdp_turbo/distributed/parallel_state.py',
 data:M+'mindspeed_llm/fsdp2/data/data_factory.py',
 opt:M+'mindspeed_llm/fsdp2/optim/optimizer.py',
 prof:M+'mindspeed_llm/fsdp2/utils/profiler.py',
 mtp:M+'mindspeed_llm/fsdp2/models/common/mtp.py',
 recompute:F+'fsdp_turbo/memory/recompute/recompute.py',
 fsdp:F+'fsdp_turbo/distributed/fully_shard_parallel/fully_shard_parallel.py',
 planControl:A+'kernels/mega_moe_bf16/mega_moe_replica_planner_control.cpp',
 planAlgo:A+'include/mega_moe_moonep_algorithm.h',
 planValidate:A+'kernels/mega_moe_bf16/mega_moe_plan_validator.cpp',
 fwCat:A+'kernels/mega_moe_bf16/include/mega_moe_bf16_catlass_impl.h',
 fwCopy:A+'kernels/mega_moe_bf16/include/mega_moe_replica_pipeline.h',
 fwSwi:A+'kernels/mega_moe_bf16/include/block_epilogue_swiglu_bf16.h',
 bwCat:A+'kernels/mega_moe_bf16/include/mega_moe_bf16_backward_catlass_impl.h',
 bwSwi:A+'kernels/mega_moe_bf16/include/block_epilogue_swiglu_backward_bf16.h',
 bwRestore:A+'kernels/mega_moe_bf16/include/mega_moe_backward_restore.h',
 bwOperand:A+'kernels/mega_moe_bf16/include/mega_moe_restore_operand.h',
 bwOwner:A+'kernels/mega_moe_bf16/include/mega_moe_owner_schedule.h',
  bwWorkspace:A+'kernels/mega_moe_bf16/include/mega_moe_workspace_info.h',
 saved:A+'ascend_deepep/saved_gate_up_mode.py',
 modelLoader:M+'mindspeed_llm/fsdp2/models/model_loader.py',
 fsdpWrap:F+'fsdp_turbo/distributed/fine_grained_fully_shard/fully_shard.py',
 fsdpState:F+'fsdp_turbo/distributed/fine_grained_fully_shard/state.py',
 fsdpParam:F+'fsdp_turbo/distributed/fine_grained_fully_shard/param_group.py',
 fsdpStrategy:F+'fsdp_turbo/distributed/fine_grained_fully_shard/strategy.py',
};
const sourcePath = name => {
 if(name.startsWith(A)) return path.join(NATIVE, name.slice(A.length));
 if(name.startsWith('experiment/')) return path.join(EXP, name.slice('experiment/'.length));
 return path.join(ROOT || REFERENCE, name);
};
const packagedPath = name => name.startsWith(A) ? '../../' + name.slice(A.length) : 'reference-src/' + name;
const filePath = key => sourcePath(files[key]);
const texts = Object.fromEntries(Object.keys(files).map(k=>[k,fs.readFileSync(filePath(k),'utf8')]));
const lines = Object.fromEntries(Object.entries(texts).map(([k,v])=>[k,v.split(/\r?\n/)]));
function find(k,needle,n=1) {
 let count=0; const i=lines[k].findIndex(s=>s.includes(needle)&&++count===n);
 if(i<0) throw new Error(`Missing source anchor ${k}: ${needle} #${n}`);
 return i+1;
}
const rows=[];
function row(id,from,needle,to,target,detail,n=1,tn=1){
 rows.push({id,from,fromLine:find(from,needle,n),to,toLine:find(to,target,tn),detail});
}
row('I01','runner','case "$METHOD" in','runner','C) dispatcher=megamoe_balanced','方法C选择megamoe_balanced、MoonEP planner与2副本槽。');
row('I02','runner','command=(python -m torch.distributed.run','bootstrap','def main():','仓库启动入口创建各rank进程，把训练入口、YAML和CLI参数交给runtime wrapper。');
row('I03','bootstrap',"runpy.run_path(str(entry), run_name='__main__')",'train','def main():','通用脚本直接传train_fsdp2.py。固定Python实验在两者之间插入whole_entry.py。');
row('I04','train','return ModelFactory.create','factory','def create(','模型创建调用；初始化分布式、数据、optimizer与Trainer的上层入口是MindSpeedAutoTrainer。');
row('I05','factory','balanced_cfg = MegaMoeBalancedConfig(','factory','ep_plan = EPPlanConfig(','planner/slots/restore/capacity/validation连同fixed_router写入EP plan。');
row('I06','factory','MindSpeedParallelEngine(','engine','self.apply_ep_modules()','engine依次应用并行转换；返回包装后的模型。');
row('I07','engine','self.model = expert_parallelize_modules(','ep','def expert_parallelize_modules(','balanced分支走普通专家EP与Expert-FSDP路径，非作者MoonEP mirror附着路径。');
row('I08','ep','distribute_experts_module(module, ep_mesh)','ep','def distribute_experts_module(','先按专家维Shard(0)切分；home=E/EP。图中表达的是EP切分，不是副本搬运。',2);
row('I09','ep','forward_fn = get_megamoe_balanced_forward_fn(','dispatcher','def get_megamoe_balanced_forward_fn(','get_dispatcher_fn显式选择balanced闭包；传入ep_group、fixed_router和config。');
row('I10','ep','module.forward = types.MethodType(experts_forward_fn, module)','dispatcher','def balanced_forward(','方法绑定使模型内self.experts(...)进入balanced_forward；原模型上层调用保持。');
row('I11','engine','self.model = expert_fully_shard_modules(','efsdp','def expert_fully_shard_modules(','EFSDP包装管理Parameter生命周期；该文件配置BF16参数、FP32 reduce及gradient divide factor。');
row('F01','trainer','outputs = self.model(**model_inputs)','model','class DeepseekV4SparseMoeBlock','模型递归调用到MoE block；不是Trainer直接调用某层MoE。');
row('F02','model','_, weights, indices = self.gate(hidden_states)','model','class DeepseekV4TopKRouter','普通TopK与hash层各有gate分支；本例fix_router随后覆盖传给experts的路由。');
row('F03','model','routed = self.experts(flat, indices, weights)','dispatcher','def balanced_forward(','flat输入、逻辑专家索引、概率；经nn.Module与FSDP hooks间接进入已绑定forward。');
row('F04','dispatcher','ids, prob = fixed_router_for_debug(','utils','def fixed_router_for_debug(','先normalize_expert_args；若fixed_router=True则覆盖gate输出，人工负载不等同自然路由语义。');
row('F05','dispatcher','pool = NativeFSDP2Pool(','bridge','class NativeFSDP2Pool:','首次静态shape建立进程共享pool；每层bridge保存权重获取闭包；不是每层复制一整套执行池。');
row('F06','dispatcher','return bridge(x, ids, prob,','bridge','def __call__(self, x, ids, prob','x[M,H] / ids[int32,M,K] / prob；bridge再次动态取得当前层unshard后的w1/w2。');
row('F07','bridge','return _ExternalExperts.apply(','bridge','class _ExternalExperts(','训练autograd入口是本文件的_ExternalExperts；不经过megaflux_device._DeviceReplica.apply。');
row('F08','bridge','self._publish(w1, w2)','bridge','def _publish(','当前stream将live Parameter detach-copy到SHMEM home。',1);
row('F09','bridge','plan = m.create_plan(ids, self.policy, self.capacity)','device','def create_plan(','申请调用epoch/lease，固定stream；planner必须返回已校验计划。');
row('F10','device','plan = planner.plan(PlanningContext','planner','def plan(self, context):','动态分派到MoonEPPlanner.plan；Native是另一个policy adapter。');
row('F11','bridge','y, _, counts, state, dispatch = m._forward_device(','full','def _forward_device(','Python MRO：Worker→Early→Full；当前真正前向实现为Full._forward_device。');
row('F12','full','y,counts,state,dispatch,_=self.op.compute_train_replica(','torch','compute_train_replica(','返回五个native张量；Python重排成y、逻辑counts视图、执行counts、state、dispatch。');
row('F13','bridge','placement = tuple(getattr(plan, n).detach().clone()','bridge',"_PLAN_TENSORS =",'冻结logical_ids、physical_ids、copies、logical_counts；避免后续层复用scratch破坏反向。');
row('F14','bridge','m.release_pending()','device','def release_pending(self):','校验当前stream和lease，释放pending。下一次调用在同一stream按序复用scratch。');
row('F15','bridge','ctx.save_for_backward(prob, *saved, header','bridge','def backward(ctx, grad):','prob/counts/state/dispatch/placement/header经save_for_backward进入本层autograd tape。');
row('F16','model','out = routed + self.shared_experts(residual)','model','class DeepseekV4SparseMoeBlock','routed结果恢复batch/seq维度后，与原共享专家分支相加。');
row('P01','device','self._epoch += 1','device','self._stream_id = torch.npu.current_stream().npu_stream','epoch单调递增；保存原stream，禁止foreign/expired/mutated lease。');
row('P02','device','plan = planner.plan(PlanningContext','planner','def plan(self, context):','调用policy adapter；context携带executor、逻辑ids、容量、epoch。');
row('P03','planner','m.workspace.online_plan_moonep(','workspace','void online_plan_moonep(','C++ Workspace接收预分配输出buffers；online_plan_impl用algorithm=-2选择MoonEP策略。');
row('P04','workspace','if(algorithm==-2) {static void* moon=dlopen','planKernel','#define mega_moe_replica_planner_control_do','EP<=8且未强制GM时，加载libascend_deepep_moonep_planner.so并调用导出符号；kernel由宏改名复用control实现。');
row('P05','planner','plan=v.validate(context,','validator','def validate(self,context','MoonEP策略输出必须经全EP计划验证；策略本身不初始化作者MoonEP权重搬运runtime。');
row('P06','validator','logical=ids.clone();physical=physical_ids.clone();cp=copies.clone()','validator','return SimpleNamespace(','返回计划持有逻辑/物理路由和copies快照；header/counts中间空间按lease复用，bridge另冻结反向placement。');
row('P07','validator','m.workspace.validate_plan_local(','workspace','void validate_plan_local(','本地device验证与摘要打包；epoch/capacity仍随调用传入。');
row('P08','validator','dist.all_gather_into_tensor(','validator','dist.all_gather_into_tensor(','全EP收集packed摘要；不是完整专家权重传输。');
row('P09','validator','m.workspace.validate_plan_global(','workspace','void validate_plan_global(','检查全局路由/容量/一致性并生成header，后续计算kernel消费。');
row('P10','validator','return SimpleNamespace(reference=','device','self._check_device_plan(plan)','certified plan包含logical_ids、physical_ids、copies、logical_counts、header、epoch、owner、max_recv。');
row('P11','device','plan._lease_tensors = tuple(','device','self._pending = True','记录tensor身份、版本、指针和capacity，防止运行期修改；同一pool串行使用。');
row('N01','full','y,counts,state,dispatch,_=self.op.compute_train_replica(','torch','compute_train_replica(','Torch自定义类调用，非torch.ops同名算子；类实例在Mixed.__init__构建。');
row('N02','torch','return ascend_deepep::ops::MegaMoeBF16Forward(','host','MegaMoeBF16Forward(','C++转发context、tensor、rank/world、SHMEM buffer及replica descriptor。');
row('N03','host','auto y = torch::empty({m, k}, x.options());','host','const auto hostTiling = detail::MakeMegaMoeHostTiling(','校验BF16/int32布局，计算tiling并分配输出和持久/瞬时workspace；pool.home权重与replica槽通过descriptor描述。',2,2);
row('N04','host','LaunchMegaMoeBF16Kernel(','launch','void LaunchMegaMoeBF16Kernel(','带replica的训练前向进入launcher；launch选择基于descriptor，不是dispatcher字符串。',2);
row('N05','launch','else if(early){static void* selected=dlopen(','launch','auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,prefetch?','early selector由Early._descriptor追加1；worker sidecar由Worker._descriptor写入descriptor[20]。');
row('N06','launch','        entry(block_dim,','kworker','#define megaflux_full_bf16_kernel_do megaflux_early_bf16_kernel_do','通过early导出名调用worker构建库中的C包装器；当前stream入队，host返回不等于设备完成。');
row('N07','kworker','#include "megaflux_full_bf16.cpp"','kfull','extern "C" ASCEND_DEEPEP_EXPORT void megaflux_full_bf16_kernel_do(','编译包含关系，不是两个设备kernel之间的运行时调用；最终导出符号由宏变成early名字。');
row('N08','kfull','megaflux_full_bf16<<<block_dim, nullptr, stream>>>','kfull','__global__ __aicore__ void megaflux_full_bf16(','C包装器用三尖括号将混合AIC/AIV kernel提交到NPU stream；源文件full名字经宏改名。');
row('N09','kfull','if(descriptor.abi!=2','kfull','op.EnableReplica(descriptor.replica);','核内校验header/epoch/capacity；有副本时配置pipeline/stripe ready，无副本时选择home tiling。');
row('N10','kfull','op.Process();','kh','__aicore__ inline void MegaMoeBF16::Process()','Process分派到saved-mode实现；包含routing/count交换、dispatch、权重stripe、GMM/SwiGLU/combine/unpermute。');
row('N11','host','return {y, expertTokenNums, trainingSavedState, dispatchOutput,','torch','compute_train_replica(','native提交后返回输出与保存状态tensor，设备工作继续按stream依赖执行。');
row('B01','trainer','loss.backward()','bridge','def backward(ctx, grad):','autograd间接触发；checkpoint重算可重新走前向，并通过saved tensor同步控制epoch。');
row('B02','bridge','def backward(ctx, grad):','bridge','prob, counts, state, dispatch, *placement_and_control = saved','从本层ctx取回前向保存状态；捕获诊断分支另有x/y，不参与常规主链。');
row('B03','bridge','w1, w2 = ctx.bridge.current_weights()','dispatcher','weights = lambda:','重新读取本层live Parameter并核对shape；不沿用被其他层覆盖的pool内容。');
row('B04','bridge','self._publish(w1, w2)','bridge','def _publish(','反向仍需刷新home与EP发布；即使前向已经复制过，本层权重也必须恢复。',2);
row('B05','bridge','replay = _SavedPlacementReplay(','bridge','class _SavedPlacementReplay:','回放前向已成功的placement，验证容量与home/slots匹配。');
row('B06','bridge','plan = m.create_plan(placement[0], replay, self.capacity)','device','def create_plan(','相同lease入口，但policy是replay，不能误画成重新调用MoonEPPlanner.plan。');
row('B07','bridge','header[2] = int(context.epoch)','bridge','def plan(self, context):','设备端clone header并更新epoch，复用前向逻辑/物理路由与copies。');
row('B08','bridge','m.workspace.restore_weights(self.shmem, plan.copies)','workspace','void restore_weights(','只有staged在host桥层显式先恢复所有副本；panel/expert将恢复交给核内ready流水。');
row('B09','bridge','dx, dp, dw1, dw2 = m.op.backward_replica(','torch','backward_replica(','输入gradY/prob/当前home权重/前向counts-state-dispatch/本次descriptor；不再依赖旧pool副本。');
row('B10','torch','ascend_deepep::ops::MegaMoeBF16Backward(','host','MegaMoeBF16Backward(','C++ binding提取四个梯度，丢弃backward workspace返回值。');
row('B11','host','LaunchMegaMoeBF16BackwardKernel(','launch','void LaunchMegaMoeBF16BackwardKernel(','host构造backward tiling、分配梯度和scratch，带descriptor启动相应执行器。');
row('B12','launch','auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,prefetch?','kbworker','#define megaflux_full_bf16_backward_ready_k_kernel_do','full-pipeline backward分支加载early SO并解析early backward导出；该库来自worker编译目标。',2);
row('B13','kbfull','op.Process();','kbh','__aicore__ inline void MegaMoeBF16Backward::Process()','early-topk路径gather/dispatch未乘p的上游梯度，panel恢复和计算按ready交错。');
row('B14','torch','return {grad_x, grad_topk_weights, grad_weight1, grad_weight2};','bridge','m.last_exec_counts = counts','C++提交后返回四个梯度tensor。');
row('B15','bridge','m.release_pending()','device','def release_pending(self):','反向释放lease，后续home梯度clone与其他设备操作按stream顺序执行。',2);
row('B16','bridge','return dx, dp, dw1[:self.home].clone(), dw2[:self.home].clone()','bridge','return (None, dx, None, dp.to(ctx.prob_dtype)','只返回home前缀并clone，明确FSDP梯度所有权；副本槽梯度已归主，不作为额外Parameter交回。');
row('B17','bridge','return (None, dx, None, dp.to(ctx.prob_dtype)','bridge','def forward(ctx, bridge, x, ids, prob, w1, w2, limit):','返回顺序严格对应apply输入：bridge无梯度，x有梯度，ids无梯度，prob/w1/w2有梯度，limit无梯度。');
row('B18','trainer','self.optimizer.step()','efsdp',"'mp_policy': MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)",'算子内owner SUM之后，框架autograd/FSDP hooks处理其reduce-scatter与缩放；Trainer执行clip、finite检查及optimizer。');
row('I12','stage','exec bash /project/tools/run-whole-a1.sh','run','command=(python -m torch.distributed.run','真实Python实验选择方法C、panel、20480容量与隔离库。');
row('I13','entry','runpy.run_path(str(entry)','train','def main():','实验harness安装采集包装并转入原训练入口。');
row('I14','engine','self.parallel_state = init_parallel_state(','mesh','def init_parallel_state(','注册普通FSDP与EP/Expert-FSDP mesh。');
row('I15','efsdp','fully_shard_fn(experts, hook_module=hook_module','fsdpWrap','def fully_shard(','本例YAML选custom。Expert-FSDP管专家Parameter，hook_module匹配相应decoder层。');
row('I16','fsdpWrap','state.init(','fsdpState','hook_module.register_forward_pre_hook(','custom fully_shard 初始化FSDPState，在匹配模块登记pre/post-forward hooks。');
row('I17','engine','WeightLoader.load(model=self.model','modelLoader','def _init_random(','本例meta + train_from_scratch，且未开pre-shard init。FSDP后to_empty与随机初始化。');
row('I18','factory','loader.create_model(model_cls=model_cls)','modelLoader','def create_model(','按实际配置在meta上构建空模型，随后才做并行装配。');
row('I19','fsdp','fully_shard_fn(module,','fsdpWrap','def fully_shard(','普通FSDP与Expert-FSDP使用同一custom实现，由各自mesh与模块计划区分。');
row('I20','train','self.trainer = Trainer(','trainer','def __init__(','模型、数据、optimizer、scheduler、checkpoint manager和monitor都已就绪，注入Trainer。');
row('F17','fsdpState','hook_module.register_forward_pre_hook(','fsdpState','self._pre_forward, prepend=True','本例hooks挂在相应decoder层。进入层时框架pre-forward负责参数准备。');
row('F18','fsdpState','self._pre_forward, prepend=True','fsdpParam','def wait_for_unshard(','继承的FSDP pre-forward触发参数准备；custom参数组在等待all-gather后执行copy-out。');
row('F19','fsdpState','def _post_forward(','fsdpState','output = self._register_pre_backward_hook(output)','层输出经过FSDP post-forward；参数组按配置reshard，输出上登记pre-backward hook。');
row('P12','planKernel','#include "mega_moe_replica_planner_control.cpp"','planControl','PLAN_START(histTimer,0)','编译时将control实现带入MoonEP planner SO。');
row('P13','planControl','PLAN_START(histTimer,0)','planControl','PLAN_START(exchangeTimer,2)','统计本地直方图与来源前缀，汇总EP计数。');
row('P14','planControl','MegaFluxMoonEP::Run(','planAlgo','inline bool Run(','计算owner到目标rank的quota，按逻辑专家剩余量分配副本。');
row('P15','planControl','PLAN_START(routesTimer,4)','planControl','PLAN_START(routesTimer,4)','根据份额和来源前缀写每条路由的物理实例。');
row('P16','workspace','ResolveValidator("megaflux_plan_validate_local_do")','planValidate','void megaflux_plan_validate_local_do(','设备本地校验生成packed摘要。');
row('P17','workspace','ResolveValidator("megaflux_plan_validate_global_do")','planValidate','void megaflux_plan_validate_global_do(','设备全局校验消费EP摘要，生成header。');
row('B19','kbworker','#include "megaflux_full_bf16_backward.cpp"','kbfull','op.Process();','编译展开后的反向实现位于full_backward.cpp。');

row('S01','engine','self.model = recompute_modules(','recompute','return checkpoint(function','方法C实验给主干与MTP目标层包装非重入checkpoint。');
row('S02','engine','self.model = fully_shard_parallel_modules(','fsdp','def fully_shard_parallel_modules(','为模型普通参数安装FSDP包装。');
row('S03','train','return DataFactory.create(','data','class MegatronDataManager(','实验YAML选择mg管理器，读取Python BIN/IDX并创建dataloader。');
row('S04','train','return OptimizerFactory.create(','opt','class OptimizerFactory:','框架根据配置创建Muon相关优化器，持有训练参数。');
row('S05','train','self.trainer.train(resume_from_checkpoint=','trainer','def train(self, resume_from_checkpoint','训练组件创建后进入训练循环。');
row('S06','trainer','batch_samples, batch_seqlens, num_items_in_batch = self.get_batch_samples_func()','trainer','loss_all = self._compute_loss(','取batch后进入training_step并计算loss。');
row('S07','model','class DeepseekV4Model(','model','mlp_output = self.mlp(','主干执行decoder层，attention之后进入MoE。');
row('S08','utils','def fixed_router_for_debug(','fixed','def fixed_routes(','11月2日实验overlay在此debug入口调用fixed_routes。左侧是当前checkout的入口，右侧是冻结实验源码。');
row('S09','model','mtp_outputs = self.mtp(','mtp','class MultiTokenPredictionBlock','MTP分支内部decoder可再次调用MoE。');
row('S10','trainer','outputs = self.model(**model_inputs)','trainer','final_mtp_loss = 0','Trainer收集LM/MTP等loss。');

row('K01','kh','void MegaMoeBF16::ProcessWithSaveMode()','kh','void MegaMoeBF16::SaveForwardState()','物理计数、索引和路由元数据为派发与反向提供地址。');
row('K02','kh','DispatchTokens<true, waitForFullExpert>','kh','void MegaMoeBF16::DispatchTokens(','按物理专家收集X及路由权重，按路径发布ready。');
row('K03','kh','MegaFluxPipeline::CopyW1Stripes(','fwCopy','void CopyW1Stripes(','副本W1/W2 stripe传输与dispatch/GEMM交错。');
row('K04','kh','MegaMoeBF16CatlassImpl::GroupMatmulSwiglu<','fwCat','void GroupMatmulSwiglu(','BF16主线进入GMM1/SwiGLU流水，依据选定路径等输入就绪。');
row('K05','fwCat','epilogueOp.template Run<SaveMode>','fwSwi','void ApplyTopkWeight(','SwiGLU先生成H，再乘p得到Hp，gate/up进入saved state。');
row('K06','kh','MegaMoeBF16CatlassImpl::GroupMatmul2','fwCat','void GroupMatmul2(','GMM2读取Hp和本地或副本W2。');
row('K07','fwCat','MegaMoeCombineImpl::CombineTokens','fwCat','MegaMoeCombineImpl::CombineTokens','把tile结果按原路由写回来源rank。');
row('K08','kh','EndGMM2Sync();','kh','void MegaMoeBF16::UnpermuteV2()','完成必要同步后按原token还原并合并TopK贡献。',2);

row('G01','kbh','// ---- P0-A:','kbh','void MegaMoeBF16Backward::DispatchTokensBackward(','gather未乘p的dY，按保存路由恢复计数和派发。');
row('G02','kbh','MegaFluxBackwardRestore::CopyExpertsPanel(','bwRestore','void CopyExpertsPanel(','AIV1搬运副本权重并按panel发布对应epoch。');
row('G03','kbh','void MegaMoeBF16Backward::GMM2GradPhase()','bwCat','void GMM2GradSwiGLUGradGM(','AIC计算dHp，AIV执行后处理；具体ready协议按路径。');
row('G04','bwCat','PanelAcquireCache p1AcquireCache','bwOperand','WaitPanel(r.flags','P1 W2 operand首次取panel检查精确epoch，可复用本AIC本专家已完成的acquire。');
row('G05','bwOperand','WaitPanel(r.flags','bwRestore','void WaitPanel(','从对应slot/projection/panel检查ready。');
row('G06','bwCat','epilogueOp.SwiGLUGrad(epiShape','bwSwi','void SwiGLUGrad(','AIV形成Hp、dp partial和dZ。');
row('G07','kbh','while (ReadGmByPassDCache(dzReady)','kbh','while (ReadGmByPassDCache(doReleased)','dZ/Hp和dW2的dO读取都退休后才进入后续阶段。');
row('G08','kbh','earlyDW1Enabled_ && GradientKey(expertIdx)','bwCat','void GMM1DW(','活动副本专家的owner与副本实例提前算dW1。');
row('G09','bwCat','MegaFluxGradient::Publish(','grad','void Publish(','梯度tile写完再发布；owner消费相同epoch。');
row('G10','kbh','GMM1GradCombineGM<WeightLayout>(dxParams','bwCat','void GMM1GradCombineGM(','dX读取W1 panel，随后将贡献回送token来源。');
row('G11','kbh','void MegaMoeBF16Backward::GMM1DWPhase()','kbh','earlyDW1Done_ &','从early cursor接续，跳过已完成的专家。');
row('G12','kbh','if(gmOutput && gradient_.ready && subBlockIdx_==1)','grad','void Reduce(','AIV1在P1后进入owner服务，AIV0走P5；两路各负责独占tile。');
row('G13','grad','MegaFluxOwnerSchedule::Task(','bwOwner','long long Task(','ratio2将较多任务交给较早到达的AIV1，保留固定贡献者求和次序。');
row('G14','grad','const int p=1-phase','grad','Wait(c,flags','先消费dW2，再消费dW1，按slot/rank读取副本贡献。');
row('G15','kbh','MEGAFLUX_STAGE_BEGIN(stageDxLocal','kbh','UnpermuteBackward();','本地与跨rank退休边界后还原token dX。',1,2);
row('G16','kbh','MEGAFLUX_STAGE_BEGIN(stageDpReuse','kbh','ReduceAndRouteTopkGrad();','归约并回送dp，再进行必要同步与清理。',1,2);

row('U01','trainer','loss.backward()','recompute','return checkpoint(function','非重入checkpoint由PyTorch按需回放层forward。');
row('U02','fsdpState','output = self._register_pre_backward_hook(output)','fsdpParam','def pre_backward(','输出hook由autograd触发；FSDP为反向恢复本层参数。');
row('U03','trainer','loss.backward()','bridge','def backward(ctx, grad):','PyTorch间接调度自定义反向。');
row('U04','fsdpParam','def post_backward(','fsdpParam','reduce_result = _call_foreach_reduce(','custom参数组收集Parameter梯度、reshard，并调用PyTorch foreach_reduce；Expert-FSDP走其自己的mesh。');
row('U05','model','out = routed + self.shared_experts(residual)','model','class DeepseekV4MLP(','shared MLP在框架计算图中独立求导，输入梯度在上游累加。');
row('U06','trainer','grad_norm = clip_grad_norm(','trainer','if math.isfinite(grad_norm):','累积完成后进行梯度裁剪和finite检查。');
row('U07','trainer','self.optimizer.step()','opt','optimizer = create_optimizer(','梯度范数有限时，Trainer 调用已创建的优化器。本实验的 optimizer=muon 进入 FSDPTurbo Muon 工厂。');
row('U08','trainer','self.lr_scheduler.step()','trainer','self.optimizer.zero_grad()','有限时随 optimizer.step 更新学习率，随后清梯度。跳过更新的分支也会清梯度。');
row('U09','trainer','self.profiler_manager.step()','prof','def stop(self):','采集窗口随step推进，训练循环结束停止profiler。');
row('U10','fsdpParam','def pre_backward(','fsdpParam','super().pre_backward(default_prefetch','custom参数组委托继承的FSDP流程进行pre-backward unshard。');

row('H01','factory','save_gmm1_output=','dispatcher','save_gmm1_output','配置随EP计划进入balanced dispatcher。');
row('H02','dispatcher','pool = NativeFSDP2Pool(','bridge','self._save_gmm1_output = validate_save_gmm1_output(','格式计入pool身份。');
row('H03','bridge','ctx.save_gmm1_output = resolve_save_gmm1_output(','saved','def resolve_save_gmm1_output(','解析调用格式，前向保存模式与反向mode匹配。');
row('H04','bridge','y, _, counts, state, dispatch = m._forward_device(','kh','void MegaMoeBF16::Process()','由launcher传递saved mode，前向按模式处理gate/up。');
row('H05','kbh','savedState_.ptrSavedGate','bwSwi','void LoadGateUpTiles(','BF16或MXFP8保存路径使用对应前向数据。');
row('H06','kbh','if (savedGateUpMode_ == MegaMoeBF16Training::kSaveNone)','kbh','void MegaMoeBF16Backward::GMM1RecomputePhase()','none走kernel内GMM1重算，和框架checkpoint是两个层级。');

row('L01','dispatcher','return bridge(x, ids, prob,','bridge','def __call__(self, x, ids, prob','dispatcher 读取 unshard 后的参数，将本层张量交给 Native bridge。');
row('L02','bridge','ctx.save_for_backward(prob, *saved, header','model','out = routed + self.shared_experts(residual)','模型消费 routed Y，并把 shared 输出相加。反向 tape 留在自定义 autograd 节点。');
row('L03','trainer','grad_norm = clip_grad_norm(','trainer','self.optimizer.zero_grad()','积累结束后裁剪梯度，有限时更新参数及学习率，然后清梯度。');
row('L04','trainer','if self.global_step >= total_steps:', 'trainer','self.profiler_manager.stop()','训练达到步数上限时跳出循环。已启用的框架 profiler 在循环外停止。');
row('L05','entry','runpy.run_path(str(entry)','entry','if TIMER:dump_frames()','训练入口返回后，采集轮导出保留的 kernel Timer 记录。');

const extra = [
 ['X01','worker','class MegaFluxBF16WorkerPipelineTrain','实际MRO入口：Worker继承Early，Early继承Full；_forward_device来自Full，create_plan/release_pending来自Device。'],
 ['X02','early','return super()._descriptor(plan)+[1]','selector=1选early launcher分支；不是用Python类名推断SO名。'],
 ['X03','worker','descriptor[20]=self.worker_sidecar.data_ptr()','Worker只扩展descriptor与ABI/feature检查；并不另写完整前向。'],
 ['X04','mixed','self.op=torch.classes.AscendDeepEP.MegaMoeBF16(local_mem_size)','创建Torch算子类与SHMEM workspace；pool的w1/w2执行存储并非optimizer拥有的Parameter。'],
 ['X05','build','cp "$BUILD/lib/libascend_deepep_megaflux_worker_bf16.so"','构建脚本将worker库复制为early库名；必须结合WORKER_NATIVE与LD_LIBRARY_PATH辨认运行库。'],
 ['X06','cmake','SOURCES kernels/mega_moe_bf16/megaflux_worker_bf16.cpp','CMake真正编译worker前反向cpp；不是另选early.cpp来猜运行实现。'],
 ['X07','kbh','MegaFluxBackwardRestore::CopyExpertsPanel','panel恢复由AIV执行；AIC使用相应ready约束权重消费。'],
 ['X08','grad','__aicore__ inline void Reduce(','gradient_pipe负责副本partial的owner归主辅助，独立于FSDP组内梯度同步。'],
 ['X09','runtimeDispatcher','def _ensure_mega_moe_runtime(','dispatcher首次调用建立/复用一套ShmemInstance与窗口；balanced显式请求window_pool_size=1，所需容量包含home/replicas/partial空间。'],
 ['X10','fsdpStrategy','class CustomFSDPStrategy','本次 YAML 的 fsdp_implementation=custom 指向 FSDPTurbo 的 fine_grained_fully_shard。该实现复用 PyTorch FSDP 的基础流程，并提供自定义 hooks 与参数组。'],
];
for(const [id,k,needle,detail] of extra) rows.push({id,from:k,fromLine:find(k,needle),to:k,toLine:find(k,needle),detail});

row('V01','bwWorkspace','ptrDX = MegaMoeBF16Training::BackwardUsesGm','bwWorkspace','ptrDO = base + offset','GM 主线的 ptrDX 与 ptrDO 同址；每专家 dW2 的 dO 读取退休后才改写 dX。');
row('V02','bwCat',"// dHpBase aliases this expert's Hp plane",'bwCat','epilogueOp.LoadDHpFromGM(dHpBase +','GM 主线先等全专家 dHpReady；每个 AIV tile 完整读入 dHp 后在相同 GM 地址写 Hp。');
row('V03','kbh','// ptrD held routed dX until every source rank completed P5.','kbh','// Batched remote dp writes must be globally complete','P5 的全部跨 rank 读者退休后，P6 才复用 ptrD 作为 dp 返回区。');

const head = dir => cp.execFileSync('git',['-C',dir,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
const heads = ROOT
 ? Object.fromEntries(['ascend_deepep_hkq','FSDPTurbo','MindSpeed-LLM','wuhan'].map(k=>[k,head(path.join(ROOT,k))]))
 : { ...require('./source-evidence.json').heads, ascend_deepep_hkq: head(NATIVE) };
const meta = {date:'2026-11-03',scope:'源码按四仓 megamoe-balanced 分支核对。主路径使用八卡方法 C、EP4 × Expert-FSDP2、E8 / Top2 和 custom FSDP。11 月 2 日采集使用独立启动包装、冻结 Python 路由、panel2 和 BF16 saved。初始化以仓库脚本说明装配，整网前向标出实验路由覆盖，反向章节的补充阅读列出可选保存格式。',
 heads,
 files:Object.fromEntries(Object.entries(files).map(([k,v])=>[k,{path:v,sha256:crypto.createHash('sha256').update(fs.readFileSync(filePath(k))).digest('hex'),lines:lines[k].length}])),rows};

const snippets = {};
for (const r of rows) {
 for (const [key,line] of [[r.from,r.fromLine],[r.to,r.toLine]]) {
  snippets[key+':'+line]=lines[key].slice(Math.max(0,line-3),line+7)
   .map((s,i)=>(String(Math.max(1,line-2)+i).padStart(5)+'  '+s).trimEnd()).join('\n');
 }
}
function copyReference(source, target) {
 if (source === target) return;
 fs.mkdirSync(path.dirname(target), { recursive: true });
 fs.copyFileSync(source, target);
}
for (const [key,info] of Object.entries(meta.files)) {
 const relative = packagedPath(info.path);
 info.href = relative.split('/').map(encodeURIComponent).join('/');
 if (!info.path.startsWith(A)) copyReference(filePath(key),path.join(OUT,relative));
}
if (ROOT) {
 for (const name of ['FSDPTurbo/LICENSE','MindSpeed-LLM/LICENSE','MindSpeed-LLM/Third_Party_Open_Source_Software_Notice']) {
  copyReference(path.join(ROOT,name),path.join(REFERENCE,name));
 }
}
fs.writeFileSync(path.join(OUT,'source-evidence.json'),JSON.stringify(meta,null,2)+'\n','utf8');
fs.writeFileSync(path.join(OUT,'source-excerpts.json'),JSON.stringify(snippets,null,2)+'\n','utf8');
