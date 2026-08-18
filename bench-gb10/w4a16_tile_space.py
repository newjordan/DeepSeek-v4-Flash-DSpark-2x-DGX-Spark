import sys
sys.path.insert(0,'/usr/local/lib/python3.12/dist-packages')
from b12x.moe.fused.w4a16 import kernel as K
SMS=48; SMEM=101376; TOPK=6
CASES=[("fc1 gate+up", 2*1024, 4096), ("fc2 down", 4096, 1024)]
ALL=sorted(set(K._SMALL_BATCH_TILE_CONFIGS)|set(K._LARGE_BATCH_TILE_CONFIGS))
for mbs in (8,16,32):
    cta_m=K._covering_count(mbs,16)
    print(f"\n=== moe_block_size={mbs} -> cta_m_blocks={cta_m} ({'LARGE' if cta_m>1 else 'SMALL'} list) ===")
    for name,n,k in CASES:
        for m in (1,8,16,36,64):
            try:
                sel=K._select_tile_config(problem_m=m,problem_n=n,problem_k=k,top_k=TOPK,
                    moe_block_size=mbs,sms=SMS,max_shared_mem=SMEM)
            except Exception as e:
                sel=f"ERR({e})"
            alts=[]
            for tk,tn,ct in ALL:
                tag=f"({tk},{tn},{ct})"
                try:
                    if not K._candidate_tile_fits(problem_n=n,problem_k=k,cta_m_blocks=cta_m,
                            tile_n=tn,tile_k=tk,cta_threads=ct,max_shared_mem=SMEM-512,
                            scale_format="e4m3_k16"):
                        alts.append(tag+":unfit"); continue
                    bps=K._determine_blocks_per_sm(problem_m=m,problem_n=n,top_k=TOPK,
                            cta_threads=ct,cta_m_blocks=cta_m,tile_n=tn,tile_k=tk,
                            uses_m_block_8=(mbs==8),sms=SMS,max_shared_mem=SMEM,
                            scale_format="e4m3_k16")
                    alts.append(f"{tag}:bps={bps}")
                except Exception as e:
                    alts.append(tag+":illegal")
            print(f"  {name:<12} m={m:<3} picked={sel}")
            print(f"               alts: {'  '.join(alts)}")
