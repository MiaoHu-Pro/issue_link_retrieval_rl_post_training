"""Audit copied CPT sources without modifying them or exporting training text.

Run from any working directory:
    python /path/to/ilr_rl_post_training/scripts/data/audit_cpt_sources.py

Uses only the Python standard library. Rewrites data/manifests/cpt_source_audit.json.
Timestamp validation here checks shape, not full calendar validity. Text overlap
is exact, whitespace-normalized, case-folded title+description within a repository.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,re
project_root=Path(__file__).resolve().parents[2]
root=project_root/'data/raw'
cutoffs={'Apache':'2019-12-19','Jira':'2018-01-26','RedHat':'2019-12-06','MongoDB':'2020-10-08','Qt':'2020-02-23','Mojang':'2021-02-06'}
result={'scope':'Full scan of the six copied issue tables and current train/test entity lists; no cleaning or CPT export performed.','datasets':{}}
for name,cutoff in cutoffs.items():
 d=root/name
 def read_ids(filename):
  ids={}; widths=Counter(); h=hashlib.sha256()
  with (d/filename).open('rb') as f:
   line=next(f);h.update(line);declared=int(line)
   for line in f:
    h.update(line);r=line.decode('utf-8').rstrip('\r\n').split('\t');widths[len(r)]+=1
    ids[r[0]]=r[1:]
  return ids,{'declared':declared,'rows':sum(widths.values()),'unique_keys':len(ids),'widths':dict(widths),'sha256':h.hexdigest()}
 train,tr=read_ids('train_entity2id.txt');test,te=read_ids('test_entity2id.txt')
 qheads=set()
 with (d/'test.txt').open() as f:
  next(f)
  for line in f:qheads.add(line.split('\t',1)[0])
 stats=Counter(); widths=Counter(); seen=set(); trainhash=Counter();testhash=set(); lengths=[]; dates=[];h=hashlib.sha256(); bad_date_examples=[]
 fpath=d/'ID_Name_Project_Type_Status_sMention_Time.txt'
 with fpath.open('rb') as f:
  for line in f:
   h.update(line);row=line.decode('utf-8').rstrip('\r\n').split('\t');widths[len(row)]+=1
   if len(row)!=8:continue
   sid,key,title,project,typ,status,desc,date=row
   stats['rows']+=1
   if key in seen:stats['duplicate_issue_key_rows']+=1
   seen.add(key)
   if not re.fullmatch(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d',date):
    stats['malformed_timestamp_shape']+=1
    if len(bad_date_examples)<3:bad_date_examples.append(date)
   dates.append(date)
   t=' '.join(title.split());de=' '.join(desc.split());normalized=(t+'\n'+de).casefold()
   digest=hashlib.sha256(normalized.encode()).hexdigest()
   intrain=key in train;intest=key in test
   if intrain:
    trainhash[digest]+=1;stats['train_rows']+=1
    lengths.append(len((t+' '+de).split()))
    stats['train_empty_title']+=not bool(t)
    stats['train_empty_description']+=not bool(de)
    stats['train_description_equals_title']+=t.casefold()==de.casefold()
    stats['train_description_placeholder']+=de.casefold() in {'none','null','nan','no name','no description'}
    stats['train_words_at_most_5']+=len((t+' '+de).split())<=5
    stats['train_issue_key_pattern']+=bool(re.search(r'\b[a-z][a-z0-9_]{1,20}-\d+\b',normalized,re.I))
    stats['train_url_pattern']+=bool(re.search(r'https?://|www\.',normalized,re.I))
    stats['train_rows_after_cutoff']+=date[:10]>cutoff
    stats['train_id_or_date_mismatch']+=train[key]!=[sid,date]
   if intest:
    testhash.add(digest);stats['test_rows']+=1
    stats['test_rows_on_or_before_cutoff']+=date[:10]<=cutoff
    stats['test_id_or_date_mismatch']+=test[key]!=[sid,date]
   if not intrain and not intest:stats['unassigned_rows']+=1
 lengths.sort()
 overlap=set(trainhash)&testhash
 result['datasets'][name]={'cutoff_inclusive_date':cutoff,'file_bytes':fpath.stat().st_size,'issue_file_sha256':h.hexdigest(),'field_counts':dict(widths),'stats':dict(stats),'min_created':min(dates),'max_created':max(dates),'train_entities':tr,'test_entities':te,'train_test_id_overlap':len(set(train)&set(test)),'train_entities_missing_text':len(set(train)-seen),'test_entities_missing_text':len(set(test)-seen),'heldout_query_heads':len(qheads),'heldout_heads_in_train_entities':len(qheads&set(train)),'train_unique_exact_texts':len(trainhash),'train_duplicate_text_extra_rows':sum(v-1 for v in trainhash.values()),'exact_text_groups_shared_train_test':len(overlap),'train_rows_matching_test_text':sum(trainhash[k] for k in overlap),'train_word_counts':{'total':sum(lengths),'p50':lengths[int(len(lengths)*.5)],'p95':lengths[int(len(lengths)*.95)],'max':max(lengths)}}
 print(name,json.dumps(result['datasets'][name]),flush=True)
out=project_root/'data/manifests/cpt_source_audit.json';out.parent.mkdir(parents=True, exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
