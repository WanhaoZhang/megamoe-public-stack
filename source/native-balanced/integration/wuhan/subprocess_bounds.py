"""Put the deadline inside Docker; killing docker exec does not kill its job."""


def bound_command(command, seconds):
    if seconds <= 0:
        raise ValueError('timeout must be positive')
    command = list(command)
    if command[:2] != ['docker', 'exec']:
        return command, seconds
    i = 2
    while i < len(command) and command[i].startswith('-'):
        if command[i] in ('-e', '--env', '-u', '--user', '-w', '--workdir'):
            if i+1 >= len(command):
                raise ValueError('missing docker exec option value')
            i += 2
        else:
            # In particular, detached exec would invalidate process completion.
            raise ValueError('unsupported docker exec option: '+command[i])
    if i+1 >= len(command):
        raise ValueError('docker exec needs a container and command')
    bounded = command[:i+1]+['timeout', '--signal=TERM', '--kill-after=20s', str(seconds)+'s']+command[i+1:]
    # GNU timeout inside the container owns the child process group. Leave time
    # for its TERM/KILL protocol before timing out the host docker client.
    return bounded, seconds+45
