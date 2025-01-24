# Internal i.AI contributors to ThemeFinder

At present we are not accepting contributions to `themefinder` as it is undergoing rapid development within i.AI.

## Testing the package locally

If you wish to install your local development version of the package to test it, you will need to install in "editable" mode:
```
pip install -e <FILE_PATH>
```
or 
```
poetry add -e <FILE_PATH>
```
where `<FILE_PATH>` is the location of your local version of `themefinder`.


## Releasing to PyPi

Creating a GitHub release will push that version of the package to TestPyPi and then PyPi.

1. Check with the Consult engineering team in Slack that it is ok to do a release.
2. Update the version number in `pyproject.toml` - note that we are using SemVer.
3. Create a [pre-release](https://github.com/i-dot-ai/themefinder/releases) by ticking the box at the bottom of the release. The release should have the tag `vX.Y.Z` where `X.Y.Z` is the version number in `pyproject.toml`.
4. Use the "Generate release notes" button to get you started on writing a suitable release note.
5. Creating the pre-release should trigger a deployment to [TestPyPi](https://test.pypi.org/project/themefinder/). Check the GitHub Actions and TestPyPi to ensure that this happens.
6. Once you're happy, go back to the pre-release and turn it into a [release](https://github.com/i-dot-ai/themefinder/releases).
7. When you publish the release, this will trigger a deployment to PyPi. You can check the GitHub actions and [PyPi](https://pypi.org/project/themefinder/) itself to confirm the deployment has worked.