from setuptools import setup

try:
	from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
	try:
		from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
	except ImportError:
		_bdist_wheel = None


cmdclass = {}

if _bdist_wheel is not None:
	class bdist_wheel(_bdist_wheel):
		def finalize_options(self):
			super().finalize_options()
			self.root_is_pure = False

		def get_tag(self):
			_, _, plat = super().get_tag()
			return "py3", "none", plat

	cmdclass["bdist_wheel"] = bdist_wheel


setup(cmdclass=cmdclass)
