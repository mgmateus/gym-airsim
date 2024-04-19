from subprocess import call

import libtmux
from libtmux.constants import PaneDirection

# server = libtmux.Server()
# session = server.find_where({"session_name": "session_test"})

# session = server.new_session(session_name="session_test", kill_session=True, attach=False)
# window = session.new_window(attach=True, window_name="session_test")
# pane1 = window.attached_pane#window.attached_pane
# pane2 = window.split(direction=PaneDirection.Above, full_window_split=True)
# window.select_layout('even-horizontal')
# pane1.send_keys('python3 settings.py')
# pane2.send_keys('ls -al')

# server.attach_session(target_session="session_test")


call(["gnome-terminal", "-x", "sh", "-c", "python3 settings.py"])
call(["gnome-terminal", "-x", "sh", "-c", "python3 ./open3d-helper/ros_api.py"])
