#!/usr/bin/env python

"""Generate a forest info file for Meraxes."""

import numpy as np
import h5py as h5
import click
from collections import OrderedDict
from tqdm import tqdm
from types import SimpleNamespace

__author__ = "Simon Mutch"
__date__ = "2017-09-12"


@click.command()
@click.argument("fname_in", type=click.Path(exists=True))
@click.argument("fname_out", type=click.Path())
def generate_forest_info(fname_in, fname_out):
    with h5.File(fname_in, "r") as fd:
        snap_groups = [k for k in fd.keys() if "Snap" in k]
        forest_ids = np.unique(np.concatenate([np.unique(fd[k]["ForestID"]) for k in snap_groups]))

        counts = OrderedDict((id, SimpleNamespace(n_halos=0,
                                                  max_n_contemp_halos=0,
                                                  n_fofs=0,
                                                  max_n_contemp_fofs=0)) for id in forest_ids)
        n_halos = np.zeros(len(snap_groups), 'int32')
        n_fofs = np.zeros(len(snap_groups), 'int32')

        # There is likely a much faster and more optimized way of doing this,
        # but we don't need to run this code very often. What's below is simple
        # and works!
        for ii, name in enumerate(tqdm(snap_groups)):
            grp = fd[name]
            ids = grp["ForestID"][:]
            is_fofs = grp["hostHaloID"][:] == -1

            n_halos[ii] = ids.shape[0]
            n_fofs[ii] = np.count_nonzero(is_fofs)

            contemps = OrderedDict((id, SimpleNamespace(halos=0, fofs=0)) for id in forest_ids)
            for id, is_fof in zip(ids, is_fofs):
                count = counts[id]
                contemp = contemps[id]
                count.n_halos += 1
                contemp.halos += 1
                if is_fof:
                    count.n_fofs += 1
                    contemp.fofs += 1

            for k, v in contemps.items():
                if v.halos > counts[k].max_n_contemp_halos:
                    counts[k].max_n_contemp_halos = v.halos
                if v.fofs > counts[k].max_n_contemp_fofs:
                    counts[k].max_n_contemp_fofs = v.fofs

    print(f"forest_ids: length={forest_ids.size}, min={forest_ids.min()}, max={forest_ids.max()}")

    def data_arr(key):
        return np.array([getattr(v, 'n_halos') for v in counts.values()], 'int32')

    with h5.File(fname_out, "w") as fd:
        cds_kwargs = dict(compression=7, shuffle=True, chunks=True)
        fd.create_dataset("n_halos", data=n_halos, **cds_kwargs)
        fd.create_dataset("n_fof_groups", data=n_fofs, **cds_kwargs)

        fd.attrs["n_snaps"] = np.array([len(snap_groups)], 'int32')[0]
        fd.attrs["n_halos_max"] = np.array([np.max(n_halos)], 'int32')[0]
        fd.attrs["n_fof_groups_max"] = np.array([np.max(n_fofs)], 'int32')[0]

        grp = fd.create_group("forests")
        grp.attrs["n_forests"] = forest_ids.size
        grp.create_dataset("forest_ids", data=forest_ids.astype('int32'), **cds_kwargs)
        grp.create_dataset("n_halos", data=data_arr('n_halos'), **cds_kwargs)
        grp.create_dataset("n_fof_groups", data=data_arr('n_fof_groups'), **cds_kwargs)
        grp.create_dataset("max_contemporaneous_halos",
                           data=data_arr('max_contemp_halos'), **cds_kwargs)
        grp.create_dataset("max_contemporaneous_fof_groups",
                           data=data_arr('max_contemp_fofs'), **cds_kwargs)


if __name__ == '__main__':
    generate_forest_info()
