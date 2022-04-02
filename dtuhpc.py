#!/usr/bin/env python3

import click
import os
import subprocess
from tabulate import tabulate

# Read ion the jobs from the hpc/jobs folder
# starting for the directory of this script
jobs_folder = os.path.join(os.path.dirname(__file__), "hpc", "jobs")
jobs_names = [
    os.path.splitext(f)[0] for f in os.listdir(jobs_folder)
]


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "name",
    type=click.Choice(jobs_names),
)
def queue(name):
    """
    Queue a job to run on the HPC.
    Pass the name of the job as an argument.

    Usage: queue-job <name>
    """
    # Check if the job is in the hpc/jobs folder starting from the directory of this script
    if not os.path.exists(
        os.path.join(os.path.dirname(__file__), "hpc", "jobs", name, name + ".py")
    ):
        print("Job not found")
        return

    # Find the queue-job.sh script in the same directory as this script
    script_path = os.path.join(os.path.dirname(__file__), "queue-job.sh")
    # Run the script with the job name
    res = os.system(f"{script_path} {name}")
    if res != 0:
        print(f"Error running {script_path} {name}")


@cli.command()
def jobs():
    """
    List all jobs on this machine that can be submitted to the HPC.
    """
    print("\n".join(jobs_names))


def check_job_folder_on_hpc(name_filter, id_filter):
    # ssh into the hpc and list the jobs folder
    res = subprocess.Popen(["ssh", "hpc", "ls", "jobs"], stdout=subprocess.PIPE)
    res.wait()
    (out, err) = res.communicate()

    jobs = out.decode("utf-8")

    # Split by the first "-" and interpret the first part as the job name and the second part as the job uuid
    jobs = [
        {"job_name": job.split("-")[0], "uuid": "-".join(job.split("-")[1:])}
        for job in jobs.split("\n")
    ]

    # Filter the jobs by name and id
    if name_filter:
        jobs = [job for job in jobs if name_filter in job["job_name"]]

    if id_filter:
        jobs = [job for job in jobs if job["uuid"].startswith(id_filter)]

    return jobs


@cli.command()
@click.option("-n", "--name-filter", help="Filter jobs by name")
@click.option("-i", "--id-filter", help="Filter jobs by id")
def state(name_filter, id_filter):
    """
    List all state jobs folder on the hpc.

    These are expected to be located on the hpc in the jobs folder.

    Usage: hpc-state [-n <name-filter>] [-i <id-filter>]
    """
    jobs = check_job_folder_on_hpc(name_filter, id_filter)
    print(
        tabulate(
            [[job["uuid"], job["job_name"]] for job in jobs],
            headers=["uuid", "job_name"],
        )
    )


def validate_uuid_filter(ctx, param, value):
    # Check the uudi filter contains only base 16 characters and dashes
    valid_chars = "0123456789abcdef-ABCDEF"
    if not all(c in valid_chars for c in value):
        raise click.BadParameter(f"Filter {value} contains invalid characters")

    if len(value) > 36:
        raise click.BadParameter(f"Filter {value} is too long ({len(value)} > 36)")

    return value


@cli.command()
@click.argument("job_id_filter", type=str, callback=validate_uuid_filter)
def download(job_id_filter):
    """
    Download the job folder from the hpc.

    <id>: String identifying the job id to cancel from the beginning of the job id.

    Usage: download-job-folder --id <id>
    """
    # Get all job_ids from the hpc
    jobs = check_job_folder_on_hpc(None, job_id_filter)

    if len(jobs) == 0:
        print(f"No job found matching filter {job_id_filter} found")
        return
    if len(jobs) > 1:
        print(f"Multiple jobs found matching filter '{job_id_filter}'")
        for job in jobs:
            print(f"\t > {job['uuid']} {(job['job_name'])}")
        return

    job = jobs[0]
    job_id = job["uuid"]
    job_name = job["job_name"] + "-" + job_id

    print("> Starting to gunzip the job folder")
    # ssh into the hpc and tar the job folder
    res = subprocess.Popen(
        [
            "ssh",
            "hpc",
            "cd",
            "jobs",
            "&&",
            "tar",
            "-czf",
            f"{job_id}.tar.gz",
            f"{job_name}",
        ],
        stdout=subprocess.DEVNULL,
    )
    res.wait()
    (out, err) = res.communicate()
    if res.returncode != 0:
        print(f"Error running tar -czf jobs/{job_id}.tar.gz jobs/{job_name}")
        return
    print("> Finished gunzipping the job folder")

    # Download the tar file
    res = subprocess.Popen(
        ["scp", f"hpc:jobs/{job_id}.tar.gz", f"{job_id}.tar.gz"],
        stdout=subprocess.DEVNULL,
    )
    (out, err) = res.communicate()
    if res.returncode != 0:
        print(f"Error running scp hpc:jobs/{job_id}.tar.gz {job_name}/{job_id}.tar.gz")
        if err:
            print(f"{err.decode('utf-8')}")
        if out:
            print(f"{out.decode('utf-8')}")
        return

    # Unzip the tar file locally into the {job_name} folder
    print("> Starting to unzip the job folder")
    res = subprocess.Popen(
        ["tar", "-xzf", f"{job_id}.tar.gz"],
        stdout=subprocess.DEVNULL,
    )
    print("> Finished unzipping the job folder")

    # Remove the tar file on the hpc
    print("> Starting to remove the tar file on the hpc")
    res = subprocess.Popen(
        ["ssh", "hpc", "rm", f"jobs/{job_id}.tar.gz"], stdout=subprocess.DEVNULL
    )
    res.wait()
    (out, err) = res.communicate()
    if res.returncode != 0:
        print(f"Error running rm jobs/{job_id}.tar.gz on hpc")
        return

    print("> Finished removing the tar file on the hpc")

    # Remove the tar file locally
    print("> Starting to remove the tar file locally")
    res = subprocess.Popen(
        ["rm", f"{job_id}.tar.gz"], stdout=subprocess.PIPE
    )
    res.wait()
    (out, err) = res.communicate()
    if res.returncode != 0:
        print(f"Error running rm {job_name}/{job_id}.tar.gz")
        return
    print("> Finished removing the tar file locally")


cli.add_command(queue)
cli.add_command(jobs)
cli.add_command(state)
cli.add_command(download)

if __name__ == "__main__":
    cli()
