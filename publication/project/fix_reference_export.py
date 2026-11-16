from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
directory=root/'references/easyinfra-6327f7bf'
expected={"cross_layer_balancer.py":"3f36113563529805f49f6cfa8f5618edffa01701","scheduler.py":"f694594fc107ccf4abed41848adf5d8ac4b1959e","LICENSE":"261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"}
records={}
for name,identity in expected.items():
    path=directory/name
    value=path.read_bytes()
    candidates=(value,value[:-1] if value.endswith(b'\n') else value)
    def blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    correct=next((data for data in candidates if blob(data)==identity),None)
    if correct is None:raise RuntimeError('Unexpected upstream bytes: '+name)
    path.write_bytes(correct)
    records[name]=dict(git_blob=identity,sha256=hashlib.sha256(correct).hexdigest(),bytes=len(correct))
(directory/'manifest.json').write_text(json.dumps(dict(repository='https://github.com/yize-wu/EasyInfra',commit='6327f7bf24086547de8f622f2a00fd763024b3e9',files=records,scope='Selected author sources, not a complete repository'),indent=2)+'\n')
print(json.dumps(records))
